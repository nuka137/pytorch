#pragma once

#include <torch/csrc/api/include/torch/types.h>
#include <torch/nn/module.h>

namespace torch {
namespace utils {
namespace hooks {


using HookFunction = std::function<Tensor(const nn::Module&, Tensor, Tensor)>;
using HooksDict = std::map<int64_t, HookFunction>;


class TORCH_API RemovableHandle final {
 public:
   RemovableHandle(HooksDict* hooks_dict);

   void remove();

   int64_t id() const;

   // serialize
   // deserialize
   //

   static int64_t next_id;

 private:
   int64_t id_ = 0;
   HooksDict* hooks_dict_ref_ = nullptr;
};


} // namespace hooks
} // namespace utils
} // namespace torch
