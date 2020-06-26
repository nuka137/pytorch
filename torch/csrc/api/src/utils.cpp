#include <torch/types.h>
#include <torch/utils.h>

namespace torch {
namespace utils {
namespace hooks {


RemovableHandle::RemovableHandle(HooksDict* hooks_dict) {
  hooks_dict_ref_ = hooks_dict;
  id_ = next_id;
  next_id++;
}

void RemovableHandle::remove() {
  HooksDict* hooks_dict = hooks_dict_ref_;
  if (hooks_dict != nullptr) {
    auto itr = hooks_dict->find(id_);
    if (itr != hooks_dict->end()) {
      hooks_dict->erase(itr);
    }
  }
}

int64_t RemovableHandle::id() const {
  return id_;
}

int64_t RemovableHandle::next_id = 0;

} // namespace hooks
} // namespace utils
} // namespace torch
