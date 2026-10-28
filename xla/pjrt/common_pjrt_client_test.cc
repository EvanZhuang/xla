/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/pjrt/common_pjrt_client.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/infer_dispatch_info.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(TransferIncarnationsTest, EmptySnapshotReturnsNullopt) {
  ASSERT_OK_AND_ASSIGN(auto transfer_incarnations,
                       TransferIncarnations(/*src_process_index=*/1,
                                            /*dst_process_index=*/2, {}));
  EXPECT_FALSE(transfer_incarnations.has_value());
}

TEST(TransferIncarnationsTest, SnapshotKeepsSourceThenDestination) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(2), IncarnationId(20)},
      {TaskId(1), IncarnationId(10)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto forward,
      TransferIncarnations(/*src_process_index=*/2, /*dst_process_index=*/1,
                           incarnations));
  ASSERT_TRUE(forward.has_value());
  EXPECT_THAT(*forward, ElementsAre(IncarnationId(20), IncarnationId(10)));

  ASSERT_OK_AND_ASSIGN(
      auto swapped,
      TransferIncarnations(/*src_process_index=*/1, /*dst_process_index=*/2,
                           incarnations));
  ASSERT_TRUE(swapped.has_value());
  EXPECT_THAT(*swapped, ElementsAre(IncarnationId(10), IncarnationId(20)));
}

TEST(TransferIncarnationsTest, SameTaskKeepsBothIncarnations) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(3), IncarnationId(7)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto transfer_incarnations,
      TransferIncarnations(/*src_process_index=*/3, /*dst_process_index=*/3,
                           incarnations));
  ASSERT_TRUE(transfer_incarnations.has_value());
  EXPECT_THAT(*transfer_incarnations,
              ElementsAre(IncarnationId(7), IncarnationId(7)));
}

TEST(TransferIncarnationsTest, MissingTaskFails) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(1), IncarnationId(10)},
  };

  EXPECT_THAT(TransferIncarnations(/*src_process_index=*/1,
                                   /*dst_process_index=*/2, incarnations),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

std::vector<Shape> MakeCanonicalDynamicParameterShapes(
    const HloModule& module) {
  std::vector<Shape> shapes =
      GetParameterShapes(module.entry_computation_layout());
  for (Shape& shape : shapes) {
    ShapeUtil::ForEachMutableSubshape(
        &shape, [](Shape* subshape, const ShapeIndex&) {
          if (subshape->IsArray() && subshape->is_dynamic() &&
              subshape->has_layout()) {
            subshape->mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(
                1024);
          }
        });
  }
  return shapes;
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsPadRealToStaticAndPreservesPadToStatic) {
  constexpr char kHlo[] = R"hlo(
HloModule pad_real_vs_pad_std

ENTRY main {
  p0 = bf16[<=16,32]{1,0} parameter(0)
  p1 = f32[<=16,32]{1,0} parameter(1)
  p2 = s32[<=16,32]{1,0} parameter(2)
  d0 = s32[] parameter(3)
  d1 = s32[] parameter(4)
  copy_p2 = s32[<=16,32]{1,0} copy(p2)
  bitcast_p2 = s32[<=16,32]{1,0} bitcast(copy_p2)
  pad_real_0 = bf16[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_std_1 = (f32[16,32]{1,0}, s32[], s32[]) custom-call(p1),
      custom_call_target="PadToStatic"
  pad_real_2 = s32[16,32]{1,0} custom-call(bitcast_p2, d0, d1),
      custom_call_target="PadRealToStatic"
  gte_1 = f32[16,32]{1,0} get-tuple-element(pad_std_1), index=0
  ROOT out = (bf16[16,32]{1,0}, f32[16,32]{1,0}, s32[16,32]{1,0})
      tuple(pad_real_0, gte_1, pad_real_2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsWideParametersReadByPlaneCustomCalls) {
  // After the TPU X64/complex rewriters, 64-bit and 128-bit `PadRealToStatic`
  // inputs are read directly by one 32-bit `PadRealToStatic` per plane.
  constexpr char kHlo[] = R"hlo(
HloModule wide_pad_real_to_static

ENTRY main {
  p0_s64 = s64[<=16,32]{1,0} parameter(0)
  p1_f64 = f64[<=16,32]{1,0} parameter(1)
  p2_c128 = c128[<=16,32]{1,0} parameter(2)
  p3_s64_std = s64[<=16,32]{1,0} parameter(3)
  d0 = s32[] parameter(4)
  d1 = s32[] parameter(5)

  pad_s64_lo = u32[16,32]{1,0} custom-call(p0_s64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":0}
  pad_s64_hi = u32[16,32]{1,0} custom-call(p0_s64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":1}
  pad_f64_lo = f32[16,32]{1,0} custom-call(p1_f64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":1}
  pad_c128 = f32[16,32]{1,0} custom-call(p2_c128, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":3}

  std_lo = u32[<=16,32]{1,0} custom-call(p3_s64_std),
      custom_call_target="X64SplitLow"
  pad_std = (u32[16,32]{1,0}, s32[], s32[]) custom-call(std_lo),
      custom_call_target="PadToStatic"
  pad_std_data = u32[16,32]{1,0} get-tuple-element(pad_std), index=0

  ROOT out = (u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0},
              f32[16,32]{1,0}, u32[16,32]{1,0})
      tuple(pad_s64_lo, pad_s64_hi, pad_f64_lo, pad_c128, pad_std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[3].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsThroughSparseCoreOffloadAsyncStartAndAsyncUpdate) {
  constexpr char kHlo[] = R"hlo(
HloModule sparsecore_offload_pad_real_to_static

%leaf_comp (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}, execution_thread="sparsecore"

%inner_async (inner_p0: s64[<=16,32], inner_d0: s32[], inner_d1: s32[]) -> u32[16,32] {
  %inner_p0 = s64[<=16,32]{1,0} parameter(0)
  %inner_d0 = s32[] parameter(1)
  %inner_d1 = s32[] parameter(2)
  ROOT %call = u32[16,32]{1,0} call(%inner_p0, %inner_d0, %inner_d1),
      to_apply=%leaf_comp
}, execution_thread="sparsecore"

%wrapper_comp (wrap_p0: s64[<=16,32], wrap_d0: s32[], wrap_d1: s32[]) -> u32[16,32] {
  %wrap_p0 = s64[<=16,32]{1,0} parameter(0)
  %wrap_d0 = s32[] parameter(1)
  %wrap_d1 = s32[] parameter(2)
  %tup = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%wrap_p0, %wrap_d0, %wrap_d1)
  %gte0 = s64[<=16,32]{1,0} get-tuple-element(%tup), index=0
  %gte1 = s32[] get-tuple-element(%tup), index=1
  %gte2 = s32[] get-tuple-element(%tup), index=2
  %start = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0}, u32[])
      async-start(%gte0, %gte1, %gte2), async_execution_thread="sparsecore",
      calls=%inner_async
  ROOT %done = u32[16,32]{1,0} async-done(%start)
}, execution_thread="sparsecore"

%outer_async (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %cc = u32[16,32]{1,0} custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="SparseCoreWrapper",
      called_computations={%wrapper_comp}
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], d0: s32[], d1: s32[]) -> u32[16,32] {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %d0 = s32[] parameter(2)
  %d1 = s32[] parameter(3)
  %offload_start = ((), (), u32[]) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async
  %offload_update = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0}, u32[])
      async-update(%offload_start, %p0, %d0, %d1)
  ROOT %offload_done = u32[16,32]{1,0} async-done(%offload_update)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsTupledEntryParameterElement) {
  constexpr char kHlo[] = R"hlo(
HloModule tupled_entry_pad_real_to_static

ENTRY main {
  arg_tuple = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0},
               (s64[<=16,32]{1,0}, f32[<=16,32]{1,0}), s32[], s32[]) parameter(0)
  p0 = s64[<=16,32]{1,0} get-tuple-element(arg_tuple), index=0
  p1 = f32[<=16,32]{1,0} get-tuple-element(arg_tuple), index=1
  nested = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0})
      get-tuple-element(arg_tuple), index=2
  nested_p0 = s64[<=16,32]{1,0} get-tuple-element(nested), index=0
  d0 = s32[] get-tuple-element(arg_tuple), index=3
  d1 = s32[] get-tuple-element(arg_tuple), index=4
  pad_real = u32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_nested = u32[16,32]{1,0} custom-call(nested_p0, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_std = (f32[16,32]{1,0}, s32[], s32[]) custom-call(p1),
      custom_call_target="PadToStatic"
  gte = f32[16,32]{1,0} get-tuple-element(pad_std), index=0
  ROOT out = (u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0})
      tuple(pad_real, pad_nested, gte)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_EQ(param_shapes.size(), 5);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(ShapeUtil::GetSubshape(param_shapes[2], {0})
                .layout()
                .dynamic_shape_metadata_prefix_bytes(),
            0);
  EXPECT_EQ(ShapeUtil::GetSubshape(param_shapes[2], {1})
                .layout()
                .dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsThroughNestedTuplesAndSingleElementTupleParameter) {
  // A single tuple entry parameter is flattened by `GetParameterShapes` into
  // one shape per top-level element, here a single nested tuple.
  constexpr char kHlo[] = R"hlo(
HloModule nested_tuple_pad_real_to_static

ENTRY main {
  arg = ((s64[<=16,32]{1,0}, s32[], s32[])) parameter(0)
  inner = (s64[<=16,32]{1,0}, s32[], s32[]) get-tuple-element(arg), index=0
  p = s64[<=16,32]{1,0} get-tuple-element(inner), index=0
  d0 = s32[] get-tuple-element(inner), index=1
  d1 = s32[] get-tuple-element(inner), index=2
  fwd = (s64[<=16,32]{1,0}, s32[]) tuple(p, d0)
  fwd2 = ((s64[<=16,32]{1,0}, s32[]), s32[]) tuple(fwd, d1)
  fwd_inner = (s64[<=16,32]{1,0}, s32[]) get-tuple-element(fwd2), index=0
  p_fwd = s64[<=16,32]{1,0} get-tuple-element(fwd_inner), index=0
  ROOT pad = u32[16,32]{1,0} custom-call(p_fwd, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_EQ(param_shapes.size(), 1);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(ShapeUtil::GetSubshape(param_shapes[0], {0})
                .layout()
                .dynamic_shape_metadata_prefix_bytes(),
            0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsParameterWithOtherUsers) {
  constexpr char kHlo[] = R"hlo(
HloModule shared_pad_real_to_static_parameter

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  pad = u32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  dyn_sum = s64[<=16,32]{1,0} add(p0, p0)
  ROOT out = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) tuple(pad, dyn_sum)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"),
                     HasSubstr("PadRealToStatic"), HasSubstr("dyn_sum"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsTupleParameterElementWithOtherUsers) {
  // The same tuple element read through a second get-tuple-element.
  constexpr char kHlo[] = R"hlo(
HloModule shared_tuple_pad_real_to_static_parameter

ENTRY main {
  arg = (f32[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  p = f32[<=16,32]{1,0} get-tuple-element(arg), index=0
  p_again = f32[<=16,32]{1,0} get-tuple-element(arg), index=0
  d0 = s32[] get-tuple-element(arg), index=1
  d1 = s32[] get-tuple-element(arg), index=2
  pad = f32[16,32]{1,0} custom-call(p, d0, d1),
      custom_call_target="PadRealToStatic"
  neg = f32[<=16,32]{1,0} negate(p_again)
  ROOT out = (f32[16,32]{1,0}, f32[<=16,32]{1,0}) tuple(pad, neg)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0 {0}"), HasSubstr("neg"))));
}

}  // namespace
}  // namespace xla
