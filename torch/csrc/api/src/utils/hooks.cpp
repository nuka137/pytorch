#pragma once

#include <torch/types.h>
#include <torch/hooks.h>

namespace torch {
namespace utils {
namespace hooks {


RemovableHandle::RemovableHandle(
    const std::map<int64_t, RemovableHandle*>* hooks_dict) {
  hooks_dict_ref = hooks_dict;
  id = next_id;
  next_id++;
}

void RemovableHandle::remove() {
  std::map<int64_t, RemovableHandle*>* hooks_dict = hooks_dict_ref;
  if (hooks_dict != nullptr && (auto itr = hooks_dict->find(id)) != hooks_dict->end()) {
    hooks_dict->erase(itr);
  }
}

RemovableHandle::next_id = 0;

} // namespace hooks
} // namespace utils
} // namespace torch
