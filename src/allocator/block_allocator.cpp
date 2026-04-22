#include "block_allocator.hpp"

BlockAllocator::BlockAllocator(int num_blocks)
    : num_blocks_(num_blocks), ref_counts_(num_blocks, 0)
{
    // Push in reverse order so block 0 is allocated first.
    for (int i = num_blocks - 1; i >= 0; --i) {
        free_list_.push(i);
    }
}

int BlockAllocator::allocate() {
    if (free_list_.empty()) {
        return -1;
    }
    int id = free_list_.top();
    free_list_.pop();
    ref_counts_[id] = 1;
    return id;
}

void BlockAllocator::free(int block_id) {
    check_id(block_id);
    if (ref_counts_[block_id] <= 0) {
        throw std::runtime_error("BlockAllocator::free called on block with ref_count <= 0");
    }
    --ref_counts_[block_id];
    if (ref_counts_[block_id] == 0) {
        free_list_.push(block_id);
    }
}

void BlockAllocator::add_ref(int block_id) {
    check_id(block_id);
    if (ref_counts_[block_id] <= 0) {
        throw std::runtime_error("BlockAllocator::add_ref called on unallocated block");
    }
    ++ref_counts_[block_id];
}

int BlockAllocator::num_free_blocks() const {
    return static_cast<int>(free_list_.size());
}

int BlockAllocator::ref_count(int block_id) const {
    check_id(block_id);
    return ref_counts_[block_id];
}

void BlockAllocator::check_id(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::out_of_range("BlockAllocator: block_id out of range");
    }
}
