// SPDX-FileCopyrightText: 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <vector>
#include "common/object_pool.h"
#include "common/spin_lock.h"

namespace Common {

template <class Traits>
class MultiLevelPageTable final {
    using Entry = typename Traits::Entry;

    static constexpr size_t L1_BITS = Traits::L1_BITS;
    static constexpr size_t L2_BITS = Traits::ADDRESS_SPACE_BITS - L1_BITS - Traits::PAGE_BITS;
    static constexpr size_t NUM_L1_ENTRIES = 1ULL << L2_BITS;
    static constexpr bool NULL_CHECK = Traits::NULL_CHECK;

    using L1Page = std::array<Entry, NUM_L1_ENTRIES>;

public:
    explicit MultiLevelPageTable(size_t chunk_size = 64)
        : top_level(1ULL << L1_BITS), page_alloc{chunk_size} {}

    ~MultiLevelPageTable() noexcept = default;

    [[nodiscard]] Entry* find(size_t page) {
        const size_t l1_page = page >> L2_BITS;
        const size_t l2_page = page & (NUM_L1_ENTRIES - 1);
        auto* block = top_level[l1_page].load(std::memory_order_acquire);
        return block ? &(*block)[l2_page] : nullptr;
    }

    [[nodiscard]] const Entry* find(size_t page) const {
        const size_t l1_page = page >> L2_BITS;
        const size_t l2_page = page & (NUM_L1_ENTRIES - 1);
        auto* block = top_level[l1_page].load(std::memory_order_acquire);
        return block ? &(*block)[l2_page] : nullptr;
    }

    void reserve(size_t start_page, size_t end_page) {
        const size_t start_l1_page = start_page >> L2_BITS;
        const size_t end_l1_page = end_page >> L2_BITS;
        for (size_t l1_page = start_l1_page; l1_page <= end_l1_page; ++l1_page) {
            GetOrCreate(l1_page);
        }
    }

    [[nodiscard]] const Entry& operator[](size_t page) const {
        const size_t l1_page = page >> L2_BITS;
        const size_t l2_page = page & (NUM_L1_ENTRIES - 1);
        auto* block = top_level[l1_page].load(std::memory_order_acquire);
        if constexpr (NULL_CHECK) {
            if (!block) [[unlikely]] {
                block = GetOrCreate(l1_page);
            }
        }
        return (*block)[l2_page];
    }

    [[nodiscard]] Entry& operator[](size_t page) {
        const size_t l1_page = page >> L2_BITS;
        const size_t l2_page = page & (NUM_L1_ENTRIES - 1);
        auto* block = top_level[l1_page].load(std::memory_order_acquire);
        if constexpr (NULL_CHECK) {
            if (!block) [[unlikely]] {
                block = GetOrCreate(l1_page);
            }
        }
        return (*block)[l2_page];
    }

private:
    // A block is published once and remains alive until table destruction.
    L1Page* GetOrCreate(size_t index) const {
        auto* block = top_level[index].load(std::memory_order_acquire);
        if (block) {
            return block;
        }
        std::lock_guard guard{top_lock};
        block = top_level[index].load(std::memory_order_relaxed);
        if (!block) {
            block = page_alloc.Create();
            top_level[index].store(block, std::memory_order_release);
        }
        return block;
    }

    mutable std::vector<std::atomic<L1Page*>> top_level;
    mutable Common::SpinLock top_lock{};
    mutable Common::ObjectPool<L1Page> page_alloc;
};

} // namespace Common
