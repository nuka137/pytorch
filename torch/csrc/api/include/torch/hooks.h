#pragma once

#include <torch/csrc/api/include/torch/types.h>

namespace torch {
namespace utils {
namespace hooks {


class TORCH_API RemovableHandle final {
 public:
   RemovableHandle(const std::map<int64_t, RemovableHandle*>* hooks_dict);

   void remove();

   // serialize
   // deserialize
   //

 private:
   static int64_t next_id;
   int64_t id = 0;
   std::map<int64_t, RemovableHandle*>* hooks_dict_ref = nullptr;
};


} // namespace hooks
} // namespace utils
} // namespace torch
