#pragma once

#include <stack>
#include <vector>
#include <stdexcept>

struct PhysicalBlock {
    int block_id;
    int ref_count;
};

class BlockAllocator {
public:
    explicit BlockAllocator(int num_blocks);

    // Returns block_id of a freshly allocated block (ref_count = 1).
    // Returns -1 if no free blocks remain.
    int allocate();

    // Decrement ref_count. Returns the block to the free list when it hits 0.
    void free(int block_id);

    // Increment ref_count for copy-on-write sharing.
    void add_ref(int block_id);

    int num_free_blocks() const;
    int ref_count(int block_id) const;

private:
    int num_blocks_;
    std::vector<int> ref_counts_;
    std::stack<int>  free_list_;

    void check_id(int block_id) const;
};
