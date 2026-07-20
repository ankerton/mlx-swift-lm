// Proof test for the 2026-07-20 fix: isRestorable/markSpeculationBoundary/
// rollbackToBoundary/commitBoundary must dispatch correctly when a concrete
// cache is held as a `KVCache` existential — that is exactly the shape every
// real call site uses (`newCache(parameters:) -> [KVCache]`, `self.caches`
// in BatchEngine, `canRestoreCache(_:)`). A test that only touches the
// concrete type would have passed even while the bug was live, so every
// assertion here goes through a `KVCache`-typed variable, never the
// concrete one, for the read that matters.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class MTPCacheDispatchProofTests: XCTestCase {

    func testKVCacheSimpleIsRestorableThroughExistential() {
        let concrete = KVCacheSimple()
        concrete.offset = 5
        let existential: KVCache = concrete

        XCTAssertTrue(
            existential.isRestorable,
            "KVCacheSimple must report isRestorable == true through the KVCache existential")

        let discarded = existential.rollbackToBoundary(discarding: 3)
        XCTAssertEqual(
            discarded, 3,
            "existential dispatch must reach KVCacheSimple.rollbackToBoundary (== trim), not the inert protocol-extension default")
        XCTAssertEqual(
            concrete.offset, 2,
            "the underlying object must actually have been mutated by the existential call")
    }

    func testMambaCacheIsRestorableThroughExistential() {
        let concrete = MambaCache()
        let existential: KVCache = concrete

        XCTAssertTrue(
            existential.isRestorable,
            "MambaCache must report isRestorable == true through the KVCache existential — "
                + "this exact assertion was FALSE before the 2026-07-20 fix")

        // No boundary marked yet — rollback must report 0 (the documented
        // "nothing to restore" case), still via the existential.
        let discardedBeforeMark = existential.rollbackToBoundary(discarding: 1)
        XCTAssertEqual(discardedBeforeMark, 0)

        existential.markSpeculationBoundary()
        // Snapshot only captures something if cache[0]/cache[1] are non-nil;
        // they're nil on a fresh cache, so the mark is a no-op recording
        // "nothing to snapshot" — rollback must still be well-defined (0),
        // not crash, through the existential.
        let discardedAfterMark = existential.rollbackToBoundary(discarding: 1)
        XCTAssertEqual(discardedAfterMark, 0)

        existential.commitBoundary()  // must not crash through the existential
    }

    func testCanRestoreCacheArrayHelperSeesRealValues() {
        // This is the exact shape `Qwen35TextModel.newCache(parameters:)` /
        // `BatchEngine.init`'s `self.caches` produce: a `[KVCache]` mixing
        // MambaCache (linear-attention layers) and KVCacheSimple (full
        // attention layers). Before the fix, `canRestoreCache` on this exact
        // array always returned `false` regardless of the concrete types
        // inside it.
        let caches: [KVCache] = [MambaCache(), KVCacheSimple(), MambaCache(), KVCacheSimple()]
        XCTAssertTrue(
            canRestoreCache(caches),
            "a [KVCache] array of only MambaCache/KVCacheSimple must be restorable — "
                + "this is the exact check that gated MTP off in production")
    }
}
