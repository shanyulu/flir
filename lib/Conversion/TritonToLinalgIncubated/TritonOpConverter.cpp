/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * Copyright (c) Microsoft Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "incubated/Conversion/TritonToLinalgIncubated/TritonOpConverter.h"
#include "incubated/Conversion/TritonToLinalgIncubated/BlockPtrAnalysis.h"
#include "incubated/Conversion/TritonToLinalgIncubated/MaskAnalysis.h"
#include "incubated/Conversion/TritonToLinalgIncubated/TritonToLinalgIncubatedPass.h"
#include "incubated/Conversion/UtilsIncubated/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#if __has_include("bishengir/Dialect/Annotation/IR/Annotation.h")
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#endif
#if __has_include("bishengir/Dialect/HFusion/IR/HFusion.h")
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#endif
#if __has_include("bishengir/Dialect/HIVM/IR/HIVM.h")
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif

#include "llvm/Config/llvm-config.h"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;
using namespace Incubated;

static const llvm::SmallVector<llvm::StringRef> libdeviceOps = {
    // Basic operations
    "__hmf_pow_fp32",
    "__hmf_div_rz_fp32",
    "__hmf_fmod_fp32",
    "__hmf_float_as_int_fp32",
    "__hmf_trunc_fp32",
    "__hmf_trunc_fp16",
    "__hmf_nearbyint_fp32",
    "__hmf_signbit_fp32",
    "__hmf_signbit_fp16",
    "__hmf_copysign_fp32",
    "__hmf_log10_fp32",
    // Trigonometric operations
    "__hmf_tanh_fp32",
    "__hmf_asin_fp32",
    "__hmf_asin_fp16",
    "__hmf_acos_fp32",
    "__hmf_acos_fp16",
    "__hmf_atan2_fp32",
    "__hmf_atan2_fp16",
    "__hmf_sinh_fp32",
    "__hmf_sinh_fp16",
    "__hmf_cosh_fp32",
    "__hmf_cosh_fp16",
    "__hmf_asinh_fp32",
    "__hmf_asinh_fp16",
    "__hmf_acosh_fp32",
    "__hmf_acosh_fp16",
    "__hmf_atanh_fp32",
    "__hmf_atanh_fp16",
    // Other operations
    "__hmf_expm1_fp32",
    "__hmf_expm1_fp16",
    "__hmf_nextafter_fp32",
    "__hmf_nextafter_fp16",
    "__hmf_hypot_fp32",
    "__hmf_hypot_fp16",
    "__hmf_cyl_bessel_i0_fp32",
    "__hmf_cyl_bessel_i0_fp16",
    "__hmf_erfinv_fp32",
    "__hmf_lgamma_fp32",
};

/**
 * Retrieves a boolean environment variable.
 * @param envVar The name of the environment variable.
 * @param defaultValue The default value to return if the variable is not set or
 * cannot be parsed.
 * @return true if the environment variable exists and its value is parsed as
 * "true", otherwise returns defaultValue. Parsing rules (case-insensitive):
 * "true" values: any non-empty string not equal to "0", "false", "no", "off" is
 * considered true. "false" values: an empty string or a string equal to any of
 * the false literals is considered false.
 */
bool getEnvBool(const char *envVar, bool defaultValue) {
  const char *val = std::getenv(envVar);
  if (val == nullptr) {
    return defaultValue; // variable not set
  }

  std::string s(val);
  // Convert to lowercase for easier comparison
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // Common false literals
  if (s.empty() || s == "0" || s == "false" || s == "no" || s == "off") {
    return false;
  }
  // All other cases (including "1", "true", "yes", "on", etc.) are considered
  // true
  return true;
}

static llvm::SmallString<kFuncNameCap>
generateUniqueFuncName(ModuleOp moduleOp, llvm::StringRef funcNameBase) {
  llvm::SmallString<kFuncNameCap> funcName = funcNameBase;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName = funcNameBase;
    funcName += ("_" + std::to_string(uniqueId++));
  }
  return funcName;
}

LogicalResult
BitcastConverter::matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Value result;
  auto loc = op.getLoc();

  if (auto dstPtrTy = dyn_cast<triton::PointerType>(op.getType())) {
    auto srcPtrTy = cast<triton::PointerType>(op.getSrc().getType());
    auto resType =
        MemRefType::get({ShapedType::kDynamic}, dstPtrTy.getPointeeType());

    auto i1Ty = rewriter.getIntegerType(1);
    auto i8Ty = rewriter.getIntegerType(8);
    bool isI1toI8 = (srcPtrTy.getPointeeType() == i1Ty) &&
                    (dstPtrTy.getPointeeType() == i8Ty);
    // handling special case: ptr<i1> -> ptr<i8>, directly forward without
    // arith.bitcast
    if (isI1toI8) {
      // TypeConverter has already converted i1 to i8 memref,
      LLVM_DEBUG({
        llvm::dbgs()
            << "[BitcastConverter] Special i1->i8 pointer bitcast. Forward "
               "without arith.bitcast. srcConvertedTy="
            << adaptor.getSrc().getType() << "\n";
      });
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    result = arith::BitcastOp::create(rewriter, loc, resType, adaptor.getSrc());
  } else {
    // handling normal case: bitcast between tensors/memrefs
    result =
        arith::BitcastOp::create(rewriter, loc, op.getType(), adaptor.getSrc());
  }
  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult
TransposeConverter::matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto src = adaptor.getSrc();
  auto res = ConverterUtils::getTransposedValue(src, op.getLoc(), rewriter,
                                                op.getOrder());
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult
YieldConverter::matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
  return success();
}

LogicalResult
AdvanceConverter::matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, BlockData> known;
  BlockDataParser::rewriteAdvanceOp(op, rewriter, known);
  return success();
}

// ToDo:
// 1. Refactor MakeTensorPtrConverter and AdvanceConverter with
// memref::ReinterpretCastOp and memref::SubViewOp.
// Use recast to describe full shape of tensor, and use subview to represent
// current block tensor.
LogicalResult MakeTensorPtrConverter::matchAndRewrite(
    triton::MakeTensorPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, BlockData> known;
  BlockDataParser::rewriteMakeTensorPtrOp(op, adaptor.getBase(), rewriter,
                                          known);
  return success();
}

LogicalResult PreciseDivConverter::matchAndRewrite(
    triton::PreciseDivFOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value opa = op.getX();
  Value opb = op.getY();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto divOp = arith::DivFOp::create(rewriter, loc, resType, opa, opb);

  rewriter.replaceOp(op, divOp);
  return success();
}

LogicalResult
SelectCanonicalizer::matchAndRewrite(arith::SelectOp op,
                                     PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // 0. Shortcut for scalars and bool type
  auto type = dyn_cast<TensorType>(op.getResult().getType());
  if (!type) {
    // do nothing non-tensor select
    return failure();
  }
  auto elementType = type.getElementType();
  if (elementType.isInteger(1)) {
    // do nothing with bool type
    return failure();
  }
  auto tensorShape = type.getShape();
  auto mask = op.getCondition();
  if (!isa<ShapedType>(mask.getType())) {
    // do nothing for scalar mask
    return failure();
  }

  // 1. Check for continuous masked loads.
  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  mlir::triton::Incubated::MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);

  if (isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return rewriter.notifyMatchFailure(
        op, "Cannot lower continuous masked selects");
  }

  // 2. Get mask position
  MaskPosition maskPos = mstate.getMaskPosition(tensorShape);
  LLVM_DEBUG({
    llvm::dbgs() << "[SelectAnalysis] MaskPosition detected: "
                 << (maskPos == MaskPosition::Head     ? "Head"
                     : maskPos == MaskPosition::Tail   ? "Tail"
                     : maskPos == MaskPosition::Middle ? "Middle"
                                                       : "Unknown")
                 << "\n";
  });

  if (maskPos == MaskPosition::Unknown) {
    mstate.eraseInsertedOps(op, rewriter);
    return failure();
  }
  auto trueTensor = op.getTrueValue();
  auto falseTensor = op.getFalseValue();

  // 3. Slice and insert out the masked part
  if (maskPos == MaskPosition::Head) {
    // Slice out the masked part of true tensor
    auto extractSliceOp = mstate.getExtractSlice(trueTensor, loc, rewriter);

    // Insert out the sliced true tensor into false tensor
    auto insertSliceOp =
        mstate.getInsertSlice(extractSliceOp, falseTensor, loc, rewriter);

    LLVM_DEBUG({
      llvm::dbgs() << "  -> Created ExtractSlice: "
                   << *extractSliceOp.getOperation() << "\n"
                   << "  -> Created InsertSlice: "
                   << *insertSliceOp.getOperation() << "\n";
    });
    rewriter.replaceOp(op, insertSliceOp);
    return success();
  }

  // For Tail or Middle positions, we need to compute inverted dimensions
  // to handle the masking logic
  SmallVector<OpFoldResult> invertOffsets;
  SmallVector<OpFoldResult> invertFalseDims;
  SmallVector<OpFoldResult> invertTrueDims;
  OpFoldResult falseDimOp;
  OpFoldResult trueDimOp;
  int valDim = -1;
  for (int i = 0; i < mstate.getRank(); ++i) {
    const auto &offVal = mstate.offsets[i];
    const auto &dimVal = mstate.dims[i];
    auto constOffVal = getConstantIntValue(offVal);
    invertOffsets.push_back(rewriter.getIndexAttr(0));
    if (constOffVal.has_value() && constOffVal.value() == 0) {
      invertFalseDims.push_back(dimVal);
      invertTrueDims.push_back(dimVal);
    } else {
      assert(valDim == -1 &&
             "The offset in only one dimension can be not zero.");
      if (!constOffVal.has_value()) {
        valDim = i;
        falseDimOp = offVal;
      }

      invertFalseDims.push_back(offVal);
      trueDimOp = addOpFoldResult(offVal, dimVal, loc, rewriter);
      invertTrueDims.push_back(trueDimOp);
    }
  }

  // Slice out the invert first masked part of false tensor
  auto falseExtractSliceOp = mstate.getExtractSlice(
      falseTensor, loc, rewriter, invertOffsets, invertFalseDims);
  // Insert out the sliced false tensor into true tensor
  auto trueInsertSliceOp =
      mstate.getInsertSlice(falseExtractSliceOp, trueTensor, loc, rewriter,
                            invertOffsets, invertFalseDims);
  // Slice out the invert first masked and masked part of inserted true tensor
  auto extractSliceOp = mstate.getExtractSlice(trueInsertSliceOp, loc, rewriter,
                                               invertOffsets, invertTrueDims);
  // Insert out the sliced true tensor into false tensor
  auto insertSliceOp =
      mstate.getInsertSlice(extractSliceOp, falseTensor, loc, rewriter,
                            invertOffsets, invertTrueDims);
  if (valDim != -1) {
    rewriter.setInsertionPointAfter(trueInsertSliceOp);
    assert(isa<Value>(falseDimOp) &&
           "Expected to be a runtime Value for dynamic dimension check.");
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value isNegative =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                              cast<Value>(falseDimOp), zeroIndex);

    Value sizeIndex =
        arith::ConstantIndexOp::create(rewriter, loc, tensorShape[valDim]);
    Value isOutOfRange =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                              cast<Value>(falseDimOp), sizeIndex);
    auto orOp = arith::OrIOp::create(rewriter, loc, isNegative, isOutOfRange);
    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{type},
                                  orOp.getResult(), true, true);

    Block *thenBlock = &ifOp.getThenRegion().front();
    rewriter.setInsertionPointToStart(thenBlock);
    scf::YieldOp::create(rewriter, loc, ValueRange{falseTensor});

    Block *elseBlock = &ifOp.getElseRegion().front();
    rewriter.setInsertionPointToStart(elseBlock);
    falseExtractSliceOp->moveBefore(elseBlock, elseBlock->begin());
    trueInsertSliceOp->moveAfter(falseExtractSliceOp);
    extractSliceOp->moveAfter(trueInsertSliceOp);
    insertSliceOp->moveAfter(extractSliceOp);

    rewriter.setInsertionPointAfter(insertSliceOp);
    scf::YieldOp::create(rewriter, loc, ValueRange{insertSliceOp.getResult()});
    rewriter.replaceOp(op, ifOp);
  } else { // static offsets
    rewriter.replaceOp(op, insertSliceOp);
  }
  LLVM_DEBUG({
    llvm::dbgs() << "  -> [invert] Created false tensor extractSlice: "
                 << *falseExtractSliceOp.getOperation() << "\n"
                 << "  -> [invert] Created true tensor insertSlice: "
                 << *trueInsertSliceOp.getOperation() << "\n"
                 << "  -> [invert] Created ExtractSlice: "
                 << *extractSliceOp.getOperation() << "\n"
                 << "  -> [invert] Created InsertSlice: "
                 << *insertSliceOp.getOperation() << "\n";
  });
  return success();
}

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly applied
 * on function arguments
 */
LogicalResult
BitcastCanonicalizer::matchAndRewrite(triton::BitcastOp bitcastOp,
                                      PatternRewriter &rewriter) const {
  Value castSrc = bitcastOp.getSrc();
  Value castRes = bitcastOp.getResult();
  Type castSrcTy = castSrc.getType();
  Type castSrcPtrTy = isa<ShapedType>(castSrcTy)
                          ? cast<ShapedType>(castSrcTy).getElementType()
                          : castSrcTy;
  if (!isa<triton::PointerType>(castSrcPtrTy))
    return failure();

  auto origBitwidth = getPointeeBitWidth(castSrc.getType());
  auto castBitwidth = getPointeeBitWidth(castRes.getType());

  if (origBitwidth == 1)
    origBitwidth = 8;
  if (castBitwidth == 1)
    castBitwidth = 8;
  if (origBitwidth != castBitwidth) {
    bitcastOp.emitError() << "Casting pointers with unmatched bitwidth!\n";
    return failure();
  }

  Operation *beforeCastOp = castSrc.getDefiningOp();
  if (beforeCastOp == nullptr) {
    return failure();
  }

  auto newRes =
      TypeSwitch<Operation *, FailureOr<Operation *>>(beforeCastOp)
          // before: addptr - bitcast - load/store
          // after: bitcast - addptr - load/store
          .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptrOp) {
            auto newCastOp =
                triton::BitcastOp::create(rewriter, bitcastOp.getLoc(),
                                          castRes.getType(), addptrOp.getPtr());
            return triton::AddPtrOp::create(
                rewriter, bitcastOp.getLoc(), castRes.getType(),
                newCastOp.getResult(), addptrOp.getOffset());
          })
          .Case<triton::SplatOp>([&](triton::SplatOp splatOp) {
            Type newCastSrcTy =
                cast<RankedTensorType>(castRes.getType()).getElementType();

            Value splatSrc = splatOp.getSrc();
            Type splatSrcTy = splatSrc.getType();
            if (auto splatSrcTensorTy = dyn_cast<RankedTensorType>(splatSrcTy))
              newCastSrcTy =
                  splatSrcTensorTy.cloneWith(std::nullopt, newCastSrcTy);
            auto newCastOp = triton::BitcastOp::create(
                rewriter, bitcastOp.getLoc(), newCastSrcTy, splatSrc);
            return triton::SplatOp::create(rewriter, bitcastOp.getLoc(),
                                           castRes.getType(), newCastOp);
          })
          // before: bitcast - bitcast
          // after(fusion optimization): bitcast
          .Case<triton::BitcastOp>([&](triton::BitcastOp prevCastOp) {
            return triton::BitcastOp::create(rewriter, bitcastOp.getLoc(),
                                             castRes.getType(),
                                             prevCastOp.getSrc());
          })
          .Default([&](Operation *op) {
            return rewriter.notifyMatchFailure(bitcastOp,
                                               "Unknown bitcast pattern");
          });
  if (succeeded(newRes)) {
    rewriter.replaceOp(bitcastOp, newRes.value());
    if (beforeCastOp->use_empty()) {
      rewriter.eraseOp(beforeCastOp);
    }
    return success();
  }
  return failure();
}

LogicalResult
FpToFpCanonicalizer::matchAndRewrite(triton::FpToFpOp op,
                                     PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value input = op.getSrc();
  auto resultType = op.getResult().getType();

  // Check if rounding mode is specified
  auto roundingMode = op.getRounding();
  if (roundingMode.has_value() &&
      roundingMode.value() != triton::RoundingMode::RTNE) {
    // Non-RTNE rounding modes (e.g., RTZ) should be handled by TritonToHFusion
    // pass Return failure here so this pattern doesn't match
    return failure();
  }

  // Handle RTNE (default) rounding mode with arith.truncf/extf
  auto srcType = cast<RankedTensorType>(input.getType());
  auto dstType = cast<RankedTensorType>(resultType);
  auto srcElemType = srcType.getElementType();
  auto dstElemType = dstType.getElementType();
  if (!isa<FloatType>(srcElemType) || !isa<FloatType>(dstElemType)) {
    return op.emitError("FpToFp expects floating point types");
  }

  unsigned srcBitwidth = srcElemType.getIntOrFloatBitWidth();
  unsigned dstBitwidth = dstElemType.getIntOrFloatBitWidth();

  // Create round_mode attribute (RINT for RTNE)
  auto roundModeAttr = hfusion::RoundModeAttr::get(rewriter.getContext(),
                                                   hfusion::RoundMode::RINT);

  if (srcBitwidth > dstBitwidth) {
    // Downcast: use arith.truncf with round_mode=rint
    auto truncOp = arith::TruncFOp::create(rewriter, loc, dstType, input);
    truncOp->setAttr("round_mode", roundModeAttr);
    rewriter.replaceOp(op, truncOp.getResult());
  } else if (srcBitwidth < dstBitwidth) {
    // Upcast: use arith.extf with round_mode=rint
    auto extOp = arith::ExtFOp::create(rewriter, loc, dstType, input);
    extOp->setAttr("round_mode", roundModeAttr);
    rewriter.replaceOp(op, extOp.getResult());
  } else {
    // Same bitwidth, should not happen but handle gracefully
    rewriter.replaceOp(op, input);
  }

  return success();
}

void rewriteUserWithNewOrder(
    mlir::OpOperand *use, PatternRewriter &rewriter,
    llvm::SmallVector<int64_t, 8> &blkShapeI64, // 8: container size
    mlir::Location &loc, llvm::ArrayRef<int32_t> &order, size_t &orderSize) {
  Operation *user = use->getOwner();
  rewriter.setInsertionPointAfter(user);
  if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
    auto loadResTy = loadOp.getResult().getType();
    auto loadResShapedTy = cast<ShapedType>(loadResTy);
    auto newLoadTy = loadResShapedTy.cloneWith(
        blkShapeI64, loadResShapedTy.getElementType());
    auto newLoadOp = triton::LoadOp::create(
        rewriter, loc, newLoadTy, loadOp->getOperands(), loadOp->getAttrs());
    newLoadOp->setAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG,
                       UnitAttr::get(rewriter.getContext()));
    rewriter.replaceOp(loadOp, newLoadOp);
    // load contiguous data then permute. thus the permute order is as
    // follows.
    SmallVector<int32_t, 8> permuteOrder; // 8: container size
    for (auto [i, v] : llvm::enumerate(order)) {
      permuteOrder.push_back(orderSize - 1 - order[i]);
    }
    auto permuteOp = triton::TransOp::create(
        rewriter, loc, newLoadOp.getResult(),
        DenseI32ArrayAttr::get(loadOp.getContext(), permuteOrder));
    newLoadOp.getResult().replaceAllUsesExcept(permuteOp.getResult(),
                                               permuteOp);
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
    // permute to contiguous then store. thus the permute order is as follows.
    SmallVector<int32_t, 8> permuteOrder; // 8: container size
    for (auto [i, v] : llvm::enumerate(order)) {
      permuteOrder.push_back(order[orderSize - 1 - i]);
    }
    auto permuteOp = triton::TransOp::create(
        rewriter, loc, storeOp.getValue(),
        DenseI32ArrayAttr::get(storeOp.getContext(), permuteOrder));
    storeOp.getValue().replaceAllUsesExcept(permuteOp.getResult(), permuteOp);
    auto newStoreOp = triton::StoreOp::create(
        rewriter, loc, storeOp.getPtr(), storeOp.getValue(), storeOp.getMask(),
        storeOp.getBoundaryCheck(), storeOp.getCache(), storeOp.getEvict());
    rewriter.replaceOp(storeOp, newStoreOp);
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
    auto advanceResPtrTy =
        cast<triton::PointerType>(advanceOp.getResult().getType());
    auto advanceResShapedTy =
        cast<ShapedType>(advanceResPtrTy.getPointeeType());
    auto newAdvanceResShapedTy = advanceResShapedTy.cloneWith(
        blkShapeI64, advanceResShapedTy.getElementType());
    auto newAdvanceResPtrTy = triton::PointerType::get(
        newAdvanceResShapedTy, advanceResPtrTy.getAddressSpace());
    auto advanceOffsets = advanceOp.getOffsets();
    llvm::SmallVector<Value, 8> newAdvanceOffsets; // 8: container size
    for (int i = orderSize - 1; i >= 0; i--) {
      newAdvanceOffsets.push_back(advanceOffsets[order[i]]);
    }
    SmallVector<OpOperand *> resUses;
    for (auto &use : advanceOp->getUses())
      resUses.push_back(&use);
    auto newAdvanceOp =
        triton::AdvanceOp::create(rewriter, loc, newAdvanceResPtrTy,
                                  advanceOp.getPtr(), newAdvanceOffsets);
    rewriter.replaceOp(advanceOp, newAdvanceOp);
    for (auto resUse : resUses)
      rewriteUserWithNewOrder(resUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
    auto initArg = use->get();
    auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
    auto resultValue = loopOp.getTiedLoopResult(use);
    iterArg.setType(initArg.getType());
    resultValue.setType(initArg.getType());
    for (auto &argUse : iterArg.getUses())
      rewriteUserWithNewOrder(&argUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
    for (auto &resUse : resultValue.getUses())
      rewriteUserWithNewOrder(&resUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
  } else if (isa<scf::YieldOp>(user)) {
    return;
  } else {
    llvm_unreachable(
        "[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
        "not used by load/store/advance op");
  }
}

void markLoadUsers(mlir::OpOperand *use, PatternRewriter &rewriter) {
  Operation *user = use->getOwner();
  if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
    loadOp->setAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG,
                    UnitAttr::get(rewriter.getContext()));
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
    return;
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
    SmallVector<OpOperand *> resUses;
    for (auto &use : advanceOp->getUses())
      resUses.push_back(&use);
    for (auto resUse : resUses)
      markLoadUsers(resUse, rewriter);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
    auto initArg = use->get();
    auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
    auto resultValue = loopOp.getTiedLoopResult(use);
    iterArg.setType(initArg.getType());
    resultValue.setType(initArg.getType());
    for (auto &argUse : iterArg.getUses())
      markLoadUsers(&argUse, rewriter);
    for (auto &resUse : resultValue.getUses())
      markLoadUsers(&resUse, rewriter);
  } else if (isa<scf::YieldOp>(user)) {
    return;
  } else {
    llvm_unreachable(
        "[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
        "not used by load/store/advance op");
  }
}

LogicalResult
MakeTensorPtrCanonicalizer::matchAndRewrite(triton::MakeTensorPtrOp op,
                                            PatternRewriter &rewriter) const {
  auto order = op.getOrder();
  auto orderSize = order.size();
  if (orderSize == 1) {
    return rewriter.notifyMatchFailure(
        op, "make_tensor_ptr's order has single value.");
  }

  bool isPermuted = false;
  for (auto [first, second] : llvm::zip(order.slice(0, orderSize - 1),
                                        order.slice(1, orderSize - 1))) {
    if (first != second + 1) {
      isPermuted = true;
      break;
    }
  }

  auto loc = op.getLoc();
  auto base = op.getBase();
  auto shape = op.getShape();
  auto strides = op.getStrides();
  auto offsets = op.getOffsets();
  auto result = op.getResult();
  SmallVector<OpOperand *> opUses;

  for (auto &use : result.getUses())
    opUses.push_back(&use);
  for (auto use : opUses)
    markLoadUsers(use, rewriter);

  if (!isPermuted) {
    return rewriter.notifyMatchFailure(
        op, "make_tensor_ptr's order is contiguous.");
  }

  llvm::SmallVector<int32_t, 8> blkShapeI32;
  llvm::SmallVector<int64_t, 8> blkShapeI64;
  auto resPtrType = cast<triton::PointerType>(result.getType());
  if (auto resShapedTy = dyn_cast<ShapedType>(resPtrType.getPointeeType())) {
    auto resBlkShape = resShapedTy.getShape();
    for (auto [i, v] : llvm::enumerate(resBlkShape)) {
      auto reverseI = orderSize - 1 - i;
      blkShapeI32.push_back(resBlkShape[order[reverseI]]);
      blkShapeI64.push_back(resBlkShape[order[reverseI]]);
    }
  }

  llvm::SmallVector<Value, 8> newShape;
  llvm::SmallVector<Value, 8> newStrides;
  llvm::SmallVector<Value, 8> newOffsets;
  for (int i = orderSize - 1; i >= 0; i--) {
    newShape.push_back(shape[order[i]]);
    newStrides.push_back(strides[order[i]]);
    newOffsets.push_back(offsets[order[i]]);
  }

  llvm::SmallVector<int, 8> contiguousOrder;
  for (int i = orderSize - 1; i >= 0; i--)
    contiguousOrder.push_back(i);

  rewriter.setInsertionPoint(op);
  auto newMakeTensorPtrOp = triton::MakeTensorPtrOp::create(
      rewriter, loc, base, ValueRange(newShape), ValueRange(newStrides),
      ValueRange(newOffsets), blkShapeI32, contiguousOrder);
  rewriter.replaceOp(op, newMakeTensorPtrOp);
  for (auto use : opUses)
    rewriteUserWithNewOrder(use, rewriter, blkShapeI64, loc, order, orderSize);
  return success();
}

LogicalResult
ReduceSingleCanonicalizer::matchAndRewrite(triton::ReduceOp reduceOp,
                                           PatternRewriter &rewriter) const {
  assert(reduceOp.getSrcs().size() <= 2 &&
         "Only reduce or reduce with index are supported");
  auto src = reduceOp.getSrcs()[0];
  auto srcType = cast<RankedTensorType>(src.getType());
  auto srcShape = srcType.getShape();
  if (llvm::any_of(srcShape, [](auto s) { return s != 1; }))
    return rewriter.notifyMatchFailure(
        reduceOp, "reduce's srcs are not all with single element");
  auto loc = reduceOp->getLoc();

  // Handle Reduce Value
  auto res = reduceOp.getResult()[0];
  Value extracted;
  if (srcType.getRank() == 1) {
    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    extracted = tensor::ExtractOp::create(rewriter, loc, src, zero.getResult())
                    .getResult();
  } else {
    auto resShape = cast<RankedTensorType>(res.getType()).getShape();
    auto collapseReassociationIndicesOptional =
        getReassociationIndicesForCollapse(srcShape, resShape);
    if (!collapseReassociationIndicesOptional.has_value()) {
      return rewriter.notifyMatchFailure(
          reduceOp, "Failure with getReassociationIndicesForCollapse call");
    }
    auto collapseReassociationIndices =
        collapseReassociationIndicesOptional.value();
    extracted = tensor::CollapseShapeOp::create(rewriter, loc, src,
                                                collapseReassociationIndices)
                    .getResult();
  }
  res.replaceAllUsesWith(extracted);

  // Handle Reduce Index
  if (reduceOp.getSrcs().size() == 1)
    return success();

  auto resIdx = reduceOp.getResult()[1];
  auto zeroI32 =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  if (srcType.getRank() == 1) {
    resIdx.replaceAllUsesWith(zeroI32);
  } else {
    auto resIdxShape = cast<RankedTensorType>(resIdx.getType()).getShape();
    auto initTensor = tensor::EmptyOp::create(rewriter, loc, resIdxShape,
                                              rewriter.getI32Type());
    auto fillOp = linalg::FillOp::create(rewriter, loc, ValueRange{zeroI32},
                                         ValueRange{initTensor});
    resIdx.replaceAllUsesWith(fillOp.getResult(0));
  }

  return success();
}

LogicalResult DenseConstantConverter::matchAndRewrite(
    arith::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto denseAttr = cast<DenseElementsAttr>(op.getValue());
  auto loc = op.getLoc();
  auto constSplatOp = arith::ConstantOp::materialize(
      rewriter, denseAttr.getSplatValue<Attribute>(),
      denseAttr.getElementType(), loc);
  auto emptyOp = tensor::EmptyOp::create(
      rewriter, loc,
      cast<RankedTensorType>(op.getResult().getType()).getShape(),
      denseAttr.getElementType());

  rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{constSplatOp},
                                              ValueRange{emptyOp});

  return success();
}

LogicalResult
MakeRangeConverter::matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto type = cast<TensorType>(op.getResult().getType());
  auto shape = type.getShape();
  auto elementType = type.getElementType();
  auto context = op.getContext();

  assert(type.getShape().size() == 1 &&
         isa<IntegerType>(type.getElementType()) &&
         type.getElementType().getIntOrFloatBitWidth() == 32 &&
         "make range can only return 1D int32 tensor");

  SmallVector<AffineMap> indexingMaps{AffineMap::get(
      /* dimCount */ 1, /* symbolCount */ 0,
      {mlir::getAffineDimExpr(0, context)}, context)};

  auto init = tensor::EmptyOp::create(rewriter, loc, shape, elementType);

  auto nestedBody = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                        ValueRange blockArgs) {
    Value index = linalg::IndexOp::create(nestedBuilder, loc, 0);
    Value res =
        arith::IndexCastOp::create(nestedBuilder, loc, elementType, index);
    linalg::YieldOp::create(nestedBuilder, loc, res);
  };

  auto linalgOp = linalg::GenericOp::create(
      rewriter, loc, op->getResultTypes(), /* operands */ ValueRange{},
      ValueRange{init}, indexingMaps, ConverterUtils::getNParallelLoopsAttrs(1),
      nestedBody);

  linalgOp->setAttr("tt.from_make_range", mlir::UnitAttr::get(context));
  linalgOp->setAttr("tt.make_range_offset",
                    mlir::IntegerAttr::get(mlir::IndexType::get(context), 0));
  linalgOp->setAttr(
      "tt.make_range_size",
      mlir::IntegerAttr::get(mlir::IndexType::get(context), shape[0]));

  int32_t startVal = op.getStartAttr().getInt();
  if (startVal == 0) {
    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }

  // Apply start offset
  Value startScaler = arith::ConstantOp::create(
      rewriter, loc,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(startVal)));
  auto startInit = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
  Value startTensor =
      linalg::FillOp::create(rewriter, loc, ValueRange{startScaler},
                             ValueRange{startInit})
          .getResult(0);
  auto addOp =
      arith::AddIOp::create(rewriter, loc, linalgOp->getResult(0), startTensor);
  rewriter.replaceOp(op, addOp);
  return success();
}

LogicalResult
SplatConverter::matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto shape = op.getType().getShape();
  auto init = tensor::EmptyOp::create(rewriter, loc, shape,
                                      op.getType().getElementType());
  if (llvm::all_of(shape, [](int64_t dim) { return dim == 1; })) {
    SmallVector<Value> idx(
        shape.size(),
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0)));
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getSrc(), init,
                                                  idx);
  } else {
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        op, ValueRange{adaptor.getSrc()}, ValueRange{init});
  }
  return success();
}

#if LLVM_VERSION_MAJOR >= 22
LogicalResult
UnsplatConverter::matchAndRewrite(triton::UnsplatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto src = adaptor.getSrc();
  auto srcType = cast<RankedTensorType>(src.getType());
  auto shape = srcType.getShape();

  // Create index constants for all dimensions (all zeros since we're extracting
  // the single element)
  SmallVector<Value> indices;
  for (int64_t dim : shape) {
    indices.push_back(
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0)));
  }

  // Extract the scalar element from the tensor
  auto elementType = srcType.getElementType();
  auto extractOp =
      tensor::ExtractOp::create(rewriter, loc, elementType, src, indices);
  rewriter.replaceOp(op, extractOp.getResult());
  return success();
}
#endif

LogicalResult
ReshapeConverter::matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto src = op.getSrc();
  auto dst = op.getResult();
  Value shape = arith::ConstantOp::create(
      rewriter, loc,
      rewriter.getI64TensorAttr(cast<ShapedType>(dst.getType()).getShape()));
  auto reshapeOp =
      tensor::ReshapeOp::create(rewriter, loc, dst.getType(), src, shape);
  rewriter.replaceOp(op, reshapeOp.getResult());
  return success();
}

LogicalResult ExpandDimsConverter::matchAndRewrite(
    triton::ExpandDimsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto src = op.getSrc();
  auto resShape = cast<ShapedType>(op.getResult().getType()).getShape();
  auto axis = op.getAxis();

  SmallVector<ReassociationIndices> reassociation;

  auto src_last_dim = resShape.size() - 2;
  auto map_func = [&](unsigned i) -> ReassociationIndices {
    if (i < axis) {
      return i == src_last_dim ? ReassociationIndices{i, i + 1}
                               : ReassociationIndices{i};
    }
    return i == axis ? ReassociationIndices{i, i + 1}
                     : ReassociationIndices{i + 1};
  };

  reassociation = llvm::to_vector(
      llvm::map_range(llvm::seq<unsigned>(0, src_last_dim + 1), map_func));

  auto expandShapeOp = tensor::ExpandShapeOp::create(
      rewriter, op.getLoc(), op.getResult().getType(), src, reassociation);
  rewriter.replaceOp(op, expandShapeOp.getResult());
  return success();
}

LogicalResult
ClampFConverter::matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto input = adaptor.getX();
  auto min_para = adaptor.getMin();
  auto max_para = adaptor.getMax();
  auto propagateNan_para = adaptor.getPropagateNan();

  if (auto input_type = dyn_cast<RankedTensorType>(input.getType())) {
    if (isa<FloatType>(min_para.getType())) {
      auto minEmptyTensor = tensor::EmptyOp::create(
          rewriter, loc, input_type.getShape(), input_type.getElementType());
      min_para = linalg::FillOp::create(rewriter, loc, ValueRange{min_para},
                                        ValueRange{minEmptyTensor})
                     .result();
    }
    if (isa<FloatType>(max_para.getType())) {
      auto maxEmptyTensor = tensor::EmptyOp::create(
          rewriter, loc, input_type.getShape(), input_type.getElementType());
      max_para = linalg::FillOp::create(rewriter, loc, ValueRange{max_para},
                                        ValueRange{maxEmptyTensor})
                     .result();
    }
  }

  if (propagateNan_para == PropagateNan::NONE) {
    auto minOp = arith::MinNumFOp::create(rewriter, loc, input, max_para);
    auto maxOp = arith::MaxNumFOp::create(rewriter, loc, min_para, minOp);
    rewriter.replaceOp(op, ValueRange{maxOp});
  } else if (propagateNan_para == PropagateNan::ALL) {
    auto minOp = arith::MinimumFOp::create(rewriter, loc, input, max_para);
    auto maxOp = arith::MaximumFOp::create(rewriter, loc, min_para, minOp);
    rewriter.replaceOp(op, ValueRange{maxOp});
  } else {
    return failure();
  }

  return success();
}

// Here convert tt.broadcast to linalg.broadcast
//
// before
// %out = tt.broadcast %in : tensor<1x4x8xf32> -> tensor<128x4x8xf32>
//
// after
// %collpased = tensor.collapse_shape %in [[0, 1], [2]] :
//                                    tensor<1x4x8xf32> into tensor<4x8xf32>
// %out = linalg.broadcast ins(%collpased : tensor<4x8xf32>)
//                         outs(%empty : tensor<128x4x8xf32>) dimensions = [0]
LogicalResult
BroadcastConverter::matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  assert(op->getNumResults() == 1 && "BroadcastOp assumes single result");

  RankedTensorType sourceType =
      cast<RankedTensorType>(adaptor.getSrc().getType());
  RankedTensorType resultType = cast<RankedTensorType>(op.getType());
  auto elementType = resultType.getElementType();
  auto loc = op.getLoc();

  auto initEmpty = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                           elementType);

  SmallVector<int64_t> broadcastDims =
      ConverterUtils::getBroadcastDims(sourceType, resultType);
  SmallVector<int64_t> unbroadcastDims =
      ConverterUtils::getUnbroadcastDims(sourceType, resultType);

  SmallVector<ReassociationIndices> collapseReassociationIndices;
  auto collapseReassociationIndicesOptional =
      getReassociationIndicesForCollapse(sourceType.getShape(),
                                         unbroadcastDims);
  if (!collapseReassociationIndicesOptional.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "Failure with getReassociationIndicesForCollapse call");
  }
  collapseReassociationIndices = collapseReassociationIndicesOptional.value();

  RankedTensorType collapseResultType =
      RankedTensorType::get(unbroadcastDims, sourceType.getElementType());

  auto collpasedOp = tensor::CollapseShapeOp::create(
      rewriter, loc, collapseResultType, adaptor.getSrc(),
      collapseReassociationIndices);

  auto broadcastOp =
      linalg::BroadcastOp::create(rewriter, loc, collpasedOp, initEmpty,
                                  rewriter.getDenseI64ArrayAttr(broadcastDims));

  rewriter.replaceOp(op, broadcastOp.getResults());
  return success();
}

// Reduce Converter
bool ReduceConverter::isReductionOpSupported(Operation *redOp) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
             arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp,
             arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
             arith::MaxUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(redOp);
}

bool ReduceConverter::isMultiReductionOpSupported(Operation *redOp) {
  return isa<arith::SubFOp, arith::SubIOp, arith::DivFOp, arith::DivSIOp,
             arith::DivUIOp, arith::RemFOp, arith::RemSIOp, arith::RemUIOp>(
      redOp);
}

Value ReduceConverter::cloneReduceOps(OpBuilder &builder, Value in, Value out,
                                      Value opIns, Value opOuts,
                                      triton::ReduceOp op) const {
  auto &reg = op->getRegion(0);
  assert(reg.getBlocks().size() == 1);
  auto &body = reg.getBlocks().front();
  auto numArguments = 2;
  assert(body.getNumArguments() == numArguments);

  Value ttIn = body.getArgument(0);
  Value ttOut = body.getArgument(1);

  IRMapping mapping;
  mapping.map(ttIn, in);
  mapping.map(ttOut, out);

  for (auto &op : body.without_terminator()) {
    builder.clone(op, mapping);
  }
  auto yield = cast<triton::ReduceReturnOp>(body.getTerminator());
  return mapping.lookup(yield->getOperand(0));
}

void ReduceConverter::checkIsNotCallOp(
    const llvm::SmallVector<Operation *> &reductionOps) const {
  llvm::for_each(reductionOps, [](Operation *op) {
    assert(!isa<triton::CallOp>(op) && "tt.call ops expected to be inlined in "
                                       "tt.reduce body in ttir building stage");
  });
}

bool ReduceConverter::isSCFOpReduce(
    const llvm::SmallVector<Operation *> &reductionOps) const {
  return (reductionOps.size() == 1 &&
          reductionOps.front()->getDialect()->getNamespace() ==
              scf::SCFDialect::getDialectNamespace());
}

bool ReduceConverter::isMultiOpReduce(
    const llvm::SmallVector<Operation *> &reductionOps) const {
  this->checkIsNotCallOp(reductionOps);

  return (reductionOps.size() > 1) ||
         (reductionOps.size() == 1 &&
          this->isMultiReductionOpSupported(reductionOps.front())) ||
         this->isSCFOpReduce(reductionOps);
}

Value ReduceConverter::computeReduceResultWithCompileFlag(
    OpBuilder &opBuilder, Location loc, Value lhs, Value rhs, Value source,
    Value initTensor, triton::ReduceOp op, bool compileOn91095Flag) const {
  // Original operation list (including all operations)
  auto originalReductionOps = this->getReductionOps(op);
  auto realReductionOps = this->getRealReductionOps(op);

  // If the size of the original operation list is greater than 1,
  // there are additional operations such as type conversion, and these
  // operations must be cloned.
  bool needClone = compileOn91095Flag || originalReductionOps.size() > 1;
  if (needClone) {
    return this->cloneReduceOps(opBuilder, lhs, rhs, source, initTensor, op);
  } else {
    assert(realReductionOps.size() == 1);
    auto rop = realReductionOps.front();
    return this->getReductionElement(lhs, rhs, loc, rop, opBuilder, false);
  }
}

LogicalResult
ReduceConverter::convertToTargetOp(triton::ReduceOp op,
                                   typename triton::ReduceOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto source = adaptor.getOperands().front();
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto elemType = sourceType.getElementType();
  auto resType = op.getResult().front().getType();
  auto loc = op.getLoc();

  // Actual operation list (filtering type conversion operations, leaving only
  // actual reduce operations)
  auto realReductionOps = this->getRealReductionOps(op);

  bool multiOpReduce = this->isMultiOpReduce(realReductionOps);
  // Reduction of arbitrary operations isn't supported because using the first
  // element across the reduction dimension requires us to iterate over a
  // subview that skips over each first element.
  if (!multiOpReduce &&
      !this->isReductionOpSupported(realReductionOps.front())) {
    if (compileOn91095Flag) {
      llvm_unreachable("All reduction cases expected to be covered");
    }
    return rewriter.notifyMatchFailure(
        op, "Only support lowering reduction with single op and limited types "
            "of reduction");
  }

  auto rop = realReductionOps.front();
  auto ropLoc = rop->getLoc();
  auto axis = op.getAxis();
  auto isVectorReduce = sourceType.getRank() == 1;

  auto constantType = elemType;

  auto accBaseConstOp =
      multiOpReduce
          ? this->getMultiOpReductionBaseConstOp(rewriter, op, ropLoc,
                                                 constantType)
          : this->getReductionBaseConstOp(rewriter, rop, constantType);

  Value initTensor;
  if (isVectorReduce) {
    auto holder = bufferization::AllocTensorOp::create(
        rewriter, loc, RankedTensorType::get({}, constantType), ValueRange{});
    initTensor =
        linalg::FillOp::create(rewriter, loc, accBaseConstOp.getResult(),
                               holder.getResult())
            .getResult(0);
  } else {
    Value init = tensor::EmptyOp::create(
        rewriter, loc, cast<RankedTensorType>(resType).getShape(),
        constantType);
    initTensor =
        linalg::FillOp::create(rewriter, loc, accBaseConstOp.getResult(), init)
            .getResult(0);
  }

  Value finalResult =
      linalg::ReduceOp::create(
          rewriter, loc, ValueRange{source}, ValueRange{initTensor},
          SmallVector<int64_t>{axis},
          [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
            assert(inputs.size() == 2);
            Value result = this->computeReduceResultWithCompileFlag(
                opBuilder, loc, inputs[0], inputs[1], source, initTensor, op,
                compileOn91095Flag);
            linalg::YieldOp::create(opBuilder, loc, result);
          })
          .getResult(0);

  if (sourceType.getRank() == 1) {
    finalResult =
        tensor::ExtractOp::create(rewriter, loc, constantType, finalResult);
  }

  rewriter.replaceOp(op, finalResult);
  return success();
}

LogicalResult ReduceConverter::convertToTargetOpExtended(
    triton::ReduceOp op, typename triton::ReduceOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto elemTypes = op.getElementTypes();

  auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
  const auto isScalarReduce = valueResultType == nullptr;

  SmallVector<Value> outputs;
  for (auto i = 0; i < op.getResult().size() && i < elemTypes.size(); i++) {
    auto result = dyn_cast<RankedTensorType>(op.getType(i));
    SmallVector<int64_t> resultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(result.getShape())};
    outputs.push_back(
        tensor::EmptyOp::create(rewriter, loc, resultShape, elemTypes[i]));
  }

  auto linalgOp = linalg::ReduceOp::create(
      rewriter, loc, adaptor.getOperands(), outputs,
      SmallVector<int64_t>{adaptor.getAxis()},
      [&](OpBuilder &b, Location loc, ValueRange inputs) {
        auto tritonReduceBlock = op.getBody();
        IRMapping mapping;
        mapping.map(tritonReduceBlock->getArguments(), inputs);

        for (auto &op : tritonReduceBlock->without_terminator()) {
          b.clone(op, mapping);
        }

        auto tritonYield = tritonReduceBlock->getTerminator();
        auto results =
            llvm::map_to_vector(tritonYield->getOperands(),
                                [&](Value val) { return mapping.lookup(val); });
        linalg::YieldOp::create(b, loc, results);
      });

  auto params = getReduceWithIndexParams(op);
  if (failed(params)) {
    return rewriter.notifyMatchFailure(op, "meaningless reduce operation");
  } else if (params->withIndexType != ReduceWithIndexType::None) {
    addReduceWithIndexAttr(*params, rewriter, linalgOp);
  }

  if (isScalarReduce) {
    SmallVector<Value> reduceResults;
    for (auto i = 0; i < linalgOp.getResults().size() && i < elemTypes.size();
         i++) {
      reduceResults.push_back(tensor::ExtractOp::create(
          rewriter, loc, elemTypes[i], linalgOp.getResults()[i], ValueRange{}));
    }
    rewriter.replaceOp(op, reduceResults);
  } else {
    rewriter.replaceOp(op, linalgOp);
  }
  return success();
}

bool ScanConverter::isReductionOpSupported(Operation *reductionOp) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp>(
      reductionOp);
}

LogicalResult
ScanConverter::convertToTargetOp(triton::ScanOp op,
                                 typename triton::ScanOp::Adaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto reductionOps = this->getReductionOps(op);
  if (reductionOps.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "No reduction op found in scan body");
  }

  llvm::SmallString<64> funcName;
  auto rop = reductionOps.front();
  if (this->isReductionOpSupported(reductionOps.front())) {
    if (isa<arith::AddFOp, arith::AddIOp>(rop)) {
      funcName = "triton_cumsum";
    } else if (isa<arith::MulFOp, arith::MulIOp>(rop)) {
      funcName = "triton_cumprod";
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(),
                               std::prev(moduleOp.getBody()->end()));

    auto loc = op.getLoc();
    auto src = adaptor.getOperands().front();
    auto resTy = op.getResult().front().getType();
    auto libFnType = rewriter.getFunctionType(
        {src.getType(), rewriter.getI32Type(), rewriter.getI1Type()}, {resTy});
    auto funcOp =
        func::FuncOp::create(rewriter, loc, funcName.str(), libFnType);

    SymbolTable symTab(moduleOp);
    auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
    if (failed(maybePrintFuncNameAttr)) {
      return op->emitError(
          "failed to create a unique func name for device_print");
    }
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

    rewriter.setInsertionPoint(op);
    auto scanAxis = op.getAxis();
    auto scanReverse = op.getReverse();
    Value axis = arith::ConstantIntOp::create(rewriter, loc, scanAxis, 32);
    Value reverseVal =
        arith::ConstantIntOp::create(rewriter, loc, scanReverse, 1);
    auto callOp = func::CallOp::create(rewriter, loc, funcOp.getSymNameAttr(),
                                       TypeRange({resTy}),
                                       ValueRange({src, axis, reverseVal}));

    rewriter.replaceOp(op, callOp);

    return success();
  } else {
    // This branch is the associative_scan op.
    bool reverse = op.getReverse();

    auto loc = op.getLoc();

    Value scanInput = op.getOperand(0);

    auto srcType = mlir::dyn_cast<RankedTensorType>(scanInput.getType());
    if (!srcType) {
      return rewriter.notifyMatchFailure(
          op, "Expected RankedTensorType input for associative_scan");
    }

    auto elementType = srcType.getElementType();
    auto shape = srcType.getShape();
    int rank = shape.size();
    int axis = op.getAxis();

    if (axis < 0 || axis >= rank) {
      return rewriter.notifyMatchFailure(op, "Invalid scan axis: " +
                                                 std::to_string(axis));
    }

    if (op->getNumRegions() < 1 || op->getRegion(0).empty()) {
      return rewriter.notifyMatchFailure(op, "Missing combine region");
    }

    OpBuilder::InsertionGuard guard(rewriter);

    auto memrefType = MemRefType::get(shape, elementType);
    Value inputMemRef =
#if LLVM_VERSION_MAJOR < 22
        bufferization::ToMemrefOp::create(rewriter, loc, memrefType, scanInput);
#else
        bufferization::ToBufferOp::create(rewriter, loc, memrefType, scanInput);
#endif
    Value outputMemRef = memref::AllocOp::create(rewriter, loc, memrefType);

    auto processDimension = [&](ArrayRef<Value> baseIdxsArray) {
      auto startInd = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
      if (reverse) {
        startInd = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                  shape[axis] - 1);
      }
      llvm::SmallVector<Value> baseIdxs(baseIdxsArray.begin(),
                                        baseIdxsArray.end());
      llvm::SmallVector<Value> firstIdx = baseIdxs;
      if (axis <= firstIdx.size()) {
        firstIdx.insert(firstIdx.begin() + axis, startInd);
      } else {
        firstIdx.push_back(startInd);
      }

      Value firstVal =
          memref::LoadOp::create(rewriter, loc, inputMemRef, firstIdx);
      memref::StoreOp::create(rewriter, loc, firstVal, outputMemRef, firstIdx);

      Value axisSize =
          memref::DimOp::create(rewriter, loc, inputMemRef, axis).getResult();
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

      Value cmp = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sgt, axisSize, one);
      auto ifOp = scf::IfOp::create(rewriter, loc, cmp, false);

      // Create a loop only when the axis size is greater than 1.
      rewriter.setInsertionPointToStart(ifOp.thenBlock());

      auto forOp = scf::ForOp::create(rewriter, loc, one, axisSize, one);
      rewriter.setInsertionPointToStart(forOp.getBody());

      Value k = forOp.getInductionVar();
      if (reverse) {
        llvm::SmallVector<Value> fixInd;
        fixInd.push_back(arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                        shape[axis] - 1)
                             .getResult());
        fixInd.push_back(k);
        auto fixIndVal = arith::SubIOp::create(rewriter, op.getLoc(), fixInd);
        k = fixIndVal.getResult();
      }
      llvm::SmallVector<Value> currIdx = baseIdxs;
      if (axis <= currIdx.size()) {
        currIdx.insert(currIdx.begin() + axis, k);
      } else {
        currIdx.push_back(k);
      }

      Value km1 = arith::SubIOp::create(rewriter, loc, k, one);
      if (reverse) {
        km1 = arith::AddIOp::create(rewriter, loc, k, one);
      }
      llvm::SmallVector<Value> prevIdx = baseIdxs;
      if (axis <= prevIdx.size()) {
        prevIdx.insert(prevIdx.begin() + axis, km1);
      } else {
        prevIdx.push_back(km1);
      }

      Value currentVal =
          memref::LoadOp::create(rewriter, loc, inputMemRef, currIdx);
      Value prevResult =
          memref::LoadOp::create(rewriter, loc, outputMemRef, prevIdx);

      Region &combineRegion = op->getRegion(0);
      Block &combineBlock = combineRegion.front();
      IRMapping mapping;
      mapping.map(combineBlock.getArgument(0), prevResult);
      mapping.map(combineBlock.getArgument(1), currentVal);

      for (Operation &innerOp : combineBlock.without_terminator()) {
        rewriter.clone(innerOp, mapping);
      }

      Operation *yieldOp = combineBlock.getTerminator();
      Value resultVal = mapping.lookup(yieldOp->getOperand(0));

      memref::StoreOp::create(rewriter, loc, resultVal, outputMemRef, currIdx);

      rewriter.setInsertionPointAfter(ifOp);
    };

    // Constructing loops for non-scanning dimensions
    llvm::SmallVector<int> nonScanDims;
    for (int i = 0; i < rank; ++i) {
      if (i != axis)
        nonScanDims.push_back(i);
    }

    createSimpleNestedLoops(rewriter, loc, outputMemRef, nonScanDims,
                            processDimension);

    rewriter.setInsertionPointAfter(op);

    mlir::Type resultType = mlir::memref::getTensorTypeFromMemRefType(
        dyn_cast<mlir::MemRefType>(outputMemRef.getType()));
    Value outputTensor =
#if LLVM_VERSION_MAJOR < 22
        bufferization::ToTensorOp::create(rewriter, loc, outputMemRef, true);
#else
        bufferization::ToTensorOp::create(rewriter, loc, resultType,
                                          outputMemRef, true);
#endif
    rewriter.replaceOp(op, outputTensor);
    return success();
  }
}

LogicalResult ScanConverter::convertToTargetOpExtended(
    triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  bool reverse = op.getReverse();

  // 1. Extract all input tensors (supports multiple inputs)
  auto operands = op->getOperands();
  if (operands.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "No input operands for extended scan");
  }

  // 2. Validate all inputs are of RankedTensorType
  llvm::SmallVector<RankedTensorType> inputTensTypes;
  for (auto operand : operands) {
    auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorTy) {
      return rewriter.notifyMatchFailure(op,
                                         "All inputs must be RankedTensorType");
    }
    inputTensTypes.push_back(tensorTy);
  }

  // 3. Validate all input tensors have the same shape (scan operation requires
  // matching input dimensions)
  auto baseShape = inputTensTypes[0].getShape();
  int rank = baseShape.size();
  int axis = op.getAxis();
  if (axis < 0 || axis >= rank) {
    return rewriter.notifyMatchFailure(op, "Invalid scan axis: " +
                                               std::to_string(axis));
  }
  for (size_t i = 1; i < inputTensTypes.size(); ++i) {
    if (inputTensTypes[i].getShape() != baseShape) {
      return rewriter.notifyMatchFailure(op,
                                         "All inputs must have the same shape");
    }
  }

  // 4. Prepare MemRefs for multiple inputs/outputs
  llvm::SmallVector<Value> inputMemRefs;
  llvm::SmallVector<Value> outputMemRefs;
  llvm::SmallVector<MemRefType> memRefTypes;
  for (size_t i = 0; i < inputTensTypes.size(); ++i) {
    auto &tensorTy = inputTensTypes[i];
    auto memRefTy =
        MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    memRefTypes.push_back(memRefTy);
    // Convert input tensors to MemRefs
    inputMemRefs.push_back(
#if LLVM_VERSION_MAJOR < 22
        bufferization::ToMemrefOp::create(rewriter, loc, memRefTy,
                                          operands[i]));
#else
        bufferization::ToBufferOp::create(rewriter, loc, memRefTy,
                                          operands[i]));
#endif
    // Allocate MemRefs for outputs
    outputMemRefs.push_back(memref::AllocOp::create(rewriter, loc, memRefTy));
  }

  // 5. Define scanning logic for multiple inputs/outputs
  LogicalResult loopResult = success();
  auto processDimension = [&](ArrayRef<Value> baseIdxsArray) {
    llvm::SmallVector<Value> baseIdxs(baseIdxsArray.begin(),
                                      baseIdxsArray.end());

    auto startInd = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    if (reverse) {
      startInd = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                baseShape[axis] - 1);
    }

    llvm::SmallVector<Value> firstIdx = baseIdxs;
    if (axis <= firstIdx.size()) {
      firstIdx.insert(firstIdx.begin() + axis, startInd);
    } else {
      firstIdx.push_back(startInd);
    }

    // 5.1 Process the first element: directly copy multiple inputs to multiple
    // outputs (initialize cumulative results)
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      Value firstVal =
          memref::LoadOp::create(rewriter, loc, inputMemRefs[i], firstIdx);
      memref::StoreOp::create(rewriter, loc, firstVal, outputMemRefs[i],
                              firstIdx);
    }

    Value axisSize =
        arith::ConstantIndexOp::create(rewriter, loc, baseShape[axis]);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                      axisSize, one);
    auto ifOp = scf::IfOp::create(rewriter, loc, cmp, false);

    // Create a loop only when the axis size is greater than 1.
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    // Use a forward loop, but handle reverse indexing inside the loop.
    auto forOp = scf::ForOp::create(rewriter, loc, one, axisSize, one);
    rewriter.setInsertionPointToStart(forOp.getBody());

    Value k = forOp.getInductionVar();

    if (reverse) {
      // Reverse scanning: Convert the forward loop index to the actual reverse
      // index. (axis_size - 1) - k
      Value axisSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, baseShape[axis]);
      Value axisSizeMinusOne =
          arith::SubIOp::create(rewriter, loc, axisSizeVal, one);
      k = arith::SubIOp::create(rewriter, loc, axisSizeMinusOne, k);
    }

    llvm::SmallVector<Value> currIdx = baseIdxs;
    if (axis <= currIdx.size()) {
      currIdx.insert(currIdx.begin() + axis, k);
    } else {
      currIdx.push_back(k);
    }

    Value prevIndex;
    if (reverse) {
      prevIndex = arith::AddIOp::create(rewriter, loc, k, one);
    } else {
      prevIndex = arith::SubIOp::create(rewriter, loc, k, one);
    }

    llvm::SmallVector<Value> prevIdx = baseIdxs;
    if (axis <= prevIdx.size()) {
      prevIdx.insert(prevIdx.begin() + axis, prevIndex);
    } else {
      prevIdx.push_back(prevIndex);
    }

    // 5.4 Load current elements and previous cumulative results
    llvm::SmallVector<Value> currentVals;
    llvm::SmallVector<Value> prevResults;
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      currentVals.push_back(
          memref::LoadOp::create(rewriter, loc, inputMemRefs[i], currIdx));
      prevResults.push_back(
          memref::LoadOp::create(rewriter, loc, outputMemRefs[i], prevIdx));
    }

    // 5.5 Bind parameters for custom reduction logic
    Region &combineRegion = op->getRegion(0);
    if (combineRegion.empty()) {
      op->emitError("Missing combine region in extended scan");
      loopResult = failure();
      return;
    }
    Block &combineBlock = combineRegion.front();
    // Validate that the number of reduction region arguments matches (number of
    // previous results + number of current elements)
    if (combineBlock.getNumArguments() != 2 * inputMemRefs.size()) {
      op->emitError("Combine region arguments mismatch with input count");
      loopResult = failure();
      return;
    }
    IRMapping mapping;
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      // Bind previous results (previous value of the i-th output) to the i-th
      // argument of the reduction region
      mapping.map(combineBlock.getArgument(i), prevResults[i]);
      // Bind current elements (current value of the i-th input) to the i+N-th
      // argument of the reduction region (N is the number of inputs)
      mapping.map(combineBlock.getArgument(i + inputMemRefs.size()),
                  currentVals[i]);
    }

    // 5.6 Clone all operations within the reduction region
    for (Operation &innerOp : combineBlock.without_terminator()) {
      rewriter.clone(innerOp, mapping);
    }

    // 5.7 Extract reduction results and store them in outputMemRef
    Operation *yieldOp = combineBlock.getTerminator();
    if (yieldOp->getNumOperands() != outputMemRefs.size()) {
      op->emitError("Combine region returns mismatch with output count");
      loopResult = failure();
      return;
    }
    for (size_t i = 0; i < outputMemRefs.size(); ++i) {
      Value resultVal = mapping.lookup(yieldOp->getOperand(i));
      memref::StoreOp::create(rewriter, loc, resultVal, outputMemRefs[i],
                              currIdx);
    }

    rewriter.setInsertionPointAfter(ifOp);
  };

  // 6. Generate nested loops for non-scan dimensions
  llvm::SmallVector<int> nonScanDims;
  for (int i = 0; i < rank; ++i) {
    if (i != axis)
      nonScanDims.push_back(i);
  }
  createSimpleNestedLoops(rewriter, loc, outputMemRefs[0], nonScanDims,
                          processDimension);

  if (failed(loopResult)) {
    return failure();
  }

  // 7. Convert multiple output MemRefs back to tensors and replace the original
  // tt.scan operation
  llvm::SmallVector<Value> outputTensors;
  for (auto outputMemRef : outputMemRefs) {
    mlir::Type resultType = mlir::memref::getTensorTypeFromMemRefType(
        dyn_cast<mlir::MemRefType>(outputMemRef.getType()));
    outputTensors.push_back(
#if LLVM_VERSION_MAJOR < 22
        bufferization::ToTensorOp::create(rewriter, loc, outputMemRef, true));
#else
        bufferization::ToTensorOp::create(rewriter, loc, resultType,
                                          outputMemRef, true));
#endif
  }
  rewriter.replaceOp(op, outputTensors);

  return success();
}

LogicalResult ExternElementwiseClOpConverter::matchAndRewrite(
    triton::ExternElementwiseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  if (!op.getPure()) {
    op->emitWarning() << "impure elementwise op!";
    return failure();
  }
  if (op.getSymbol().contains("__hmf_")) {
    // 1. get or create the declaration of external elementwise function
    Type dstTy = op.getResult().getType();
    bool isDstScalar = !isa<RankedTensorType>(dstTy);
    Type dstElemTy =
        isDstScalar ? dstTy : cast<RankedTensorType>(dstTy).getElementType();
    SmallVector<Type, 4> srcElemTys;
    SmallVector<Value, 4> srcs;
    for (auto src : op.getSrcs()) {
      if (!isa<RankedTensorType>(src.getType())) {
        src = tensor::FromElementsOp::create(
            rewriter, op.getLoc(),
            RankedTensorType::get({(int64_t)1}, src.getType()), src);
      }
      srcs.push_back(src);
      srcElemTys.push_back(
          cast<RankedTensorType>(src.getType()).getElementType());
    }
    FunctionType elemFuncType =
        FunctionType::get(rewriter.getContext(), srcElemTys, {dstElemTy});
    auto mod = SymbolTable::getNearestSymbolTable(op);
    auto extFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(mod, op.getSymbol()));
    // std::string symbol = op.getSymbol().str();
    bool is_libdevice = llvm::is_contained(libdeviceOps, op.getSymbol()) &&
                        getEnvBool("TRITON_ENABLE_LIBDEVICE_SIMT", false);
    if (!extFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&mod->getRegion(0).front());
      extFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                     op.getSymbol(), elemFuncType);
      extFunc.setPrivate();
      extFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                       UnitAttr::get(rewriter.getContext()));
      // set coreType for external func, otherwise InferFuncCoreTypePass will
      // fail
      if (is_libdevice) {
        hivm::TFuncCoreType e = hivm::TFuncCoreType::AIV;
        extFunc->setAttr(
            hivm::TFuncCoreTypeAttr::name,
            hivm::TFuncCoreTypeAttr::get(extFunc->getContext(), e));
      }
    }
    assert(isa<FunctionOpInterface>(
        SymbolTable::lookupSymbolIn(mod, op.getSymbol())));
    // 2. prepare the output tensor
    Value output;
    if (isDstScalar) {
      dstTy = RankedTensorType::get({(int64_t)1}, dstElemTy);
    }
    bool found = false;
    for (Value v : srcs) {
      if (v.getType() == dstTy) {
        found = true;
        output = v;
        break;
      }
    }
    if (!found) {
      output = tensor::EmptyOp::create(rewriter, op.getLoc(),
                                       cast<RankedTensorType>(dstTy).getShape(),
                                       dstElemTy);
    }

    if (is_libdevice) {
      auto srcType = cast<RankedTensorType>(srcs[0].getType());
      SmallVector<Value> dimSizes;
      int64_t rank = srcType.getRank();
      for (int i = 0; i < rank; ++i) {
        if (srcType.isDynamicDim(i)) {
          auto dimOp = tensor::DimOp::create(rewriter, loc, srcs[0], i);
          dimSizes.push_back(dimOp);
        } else {
          auto constOp = arith::ConstantIndexOp::create(rewriter, loc,
                                                        srcType.getDimSize(i));
          dimSizes.push_back(constOp);
        }
      }
      // building nested loops by recursion
      std::function<Value(OpBuilder &, Location, SmallVector<Value>, Value)>
          buildLoops = [&](OpBuilder &b, Location loc,
                           SmallVector<Value> indices, Value acc) -> Value {
        int64_t dim = indices.size();
        if (dim == rank) {
          // innermost loop
          SmallVector<Value> elemVals;
          for (auto src : srcs) {
            auto extract = tensor::ExtractOp::create(b, loc, src, indices);
            elemVals.push_back(extract);
          }
          auto call =
              func::CallOp::create(b, loc, op.getSymbol(), dstElemTy, elemVals);
          auto insert =
              tensor::InsertOp::create(b, loc, call.getResult(0), acc, indices);
          return insert;
        } else {
          Value lower = arith::ConstantIndexOp::create(b, loc, 0);
          Value upper = dimSizes[dim];
          Value step = arith::ConstantIndexOp::create(b, loc, 1);
          auto loop =
              scf::ForOp::create(b, loc, lower, upper, step, ValueRange{acc});
          Block *body = loop.getBody();
          OpBuilder innerBuilder = OpBuilder::atBlockBegin(body);
          SmallVector<Value> newIndices = indices;
          newIndices.push_back(loop.getInductionVar());
          Value innerAcc = loop.getRegionIterArgs()[0];
          Value updatedAcc =
              buildLoops(innerBuilder, loc, newIndices, innerAcc);
          scf::YieldOp::create(innerBuilder, loc, updatedAcc);
          return loop.getResult(0);
        }
      };

      Value result = buildLoops(rewriter, loc, {}, output);
      if (isDstScalar) {
        SmallVector<Value> zeroIndices(
            rank, arith::ConstantIndexOp::create(rewriter, loc, 0));
        auto extract =
            tensor::ExtractOp::create(rewriter, loc, result, zeroIndices);
        rewriter.replaceOp(op, extract);
      } else {
        rewriter.replaceOp(op, result);
      }
      return success();
    }
    // 3. create the linalg.map op
    auto mapOp = linalg::MapOp::create(
        rewriter, loc,
        /*inputs=*/srcs,
        /*init=*/output,
        /*bodyBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto elemOp = func::CallOp::create(builder, loc,
                                             /*name=*/op.getSymbol(),
                                             /*resultType=*/dstElemTy,
                                             /*operands=*/regionArgs);
          linalg::YieldOp::create(builder, loc, elemOp->getResults());
        });
    if (isDstScalar) {
      // need to convert tensor back to scalar
      auto indexType = rewriter.getIndexType();
      Value zeroConstant = arith::ConstantOp::create(
          rewriter, loc, indexType, rewriter.getIntegerAttr(indexType, 0));
      auto extractOp = tensor::ExtractOp::create(
          rewriter, loc, mapOp.getResults()[0], zeroConstant);
      rewriter.replaceOp(op, extractOp);
    } else {
      rewriter.replaceOp(op, mapOp);
    }
    return success();
  }
  return failure();
}

LogicalResult UnrealizedCastConverter::matchAndRewrite(
    UnrealizedConversionCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult
JoinConverter::matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value opa = op.getLhs();
  Value opb = op.getRhs();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  Value emptyOp = tensor::EmptyOp::create(rewriter, loc, resType.getShape(),
                                          resType.getElementType());

  auto shape = dyn_cast<RankedTensorType>(opa.getType()).getShape();
  auto sizes = llvm::map_to_vector(shape, [&](int64_t t) {
    return OpFoldResult(rewriter.getI64IntegerAttr(t));
  });
  sizes.push_back(rewriter.getI64IntegerAttr(1));

  int64_t rank = resType.getRank();

  // Set last dimension stride to 2 in layout
  // As last dimension size is always 1, last dimension stride here could be
  // either 1 or 2, while stride `2` could carry interleave trait and it's
  // convenient for next lower.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  strides.back() = rewriter.getIndexAttr(2);

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));

  auto insert0 = tensor::InsertSliceOp::create(rewriter, loc, opa, emptyOp,
                                               offsets, sizes, strides);

  offsets.back() = rewriter.getIndexAttr(1);
  auto insert1 = tensor::InsertSliceOp::create(rewriter, loc, opb, insert0,
                                               offsets, sizes, strides);
  rewriter.replaceOp(op, insert1);
  return success();
}

LogicalResult
CatConverter::matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Value opa = op.getLhs();
  Value opb = op.getRhs();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resType || resType.getRank() != 1) {
    return rewriter.notifyMatchFailure(op, "only support 1D cat");
  }

  auto inputTypeA = dyn_cast<RankedTensorType>(opa.getType());
  auto inputTypeB = dyn_cast<RankedTensorType>(opb.getType());
  if (!inputTypeA || !inputTypeB || inputTypeA.getRank() != 1 ||
      inputTypeB.getRank() != 1) {
    return rewriter.notifyMatchFailure(op, "inputs must be 1D tensors");
  }

  int64_t sizeA = inputTypeA.getShape()[0];
  int64_t sizeB = inputTypeB.getShape()[0];

  // Only handle the case where both inputs have size 1 (i.e., scalar-like)
  if (sizeA == 1 && sizeB == 1) {
    // Use scalar extract + insert
    auto emptyOp = tensor::EmptyOp::create(rewriter, loc, resType.getShape(),
                                           resType.getElementType());

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    Value scalarA = tensor::ExtractOp::create(rewriter, loc, opa, zero);
    Value scalarB = tensor::ExtractOp::create(rewriter, loc, opb, zero);

    Value inserted0 =
        tensor::InsertOp::create(rewriter, loc, scalarA, emptyOp, zero);
    Value inserted1 =
        tensor::InsertOp::create(rewriter, loc, scalarB, inserted0, one);

    rewriter.replaceOp(op, inserted1);
    return success();
  }

  // General case: use tensor.insert_slice
  auto emptyOp = tensor::EmptyOp::create(rewriter, loc, resType.getShape(),
                                         resType.getElementType());

  auto rank = resType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  auto inputType = dyn_cast<RankedTensorType>(opa.getType());

  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      });

  auto insert0 = tensor::InsertSliceOp::create(rewriter, loc, opa, emptyOp,
                                               offsets, sizes, strides);

  offsets[0] =
      rewriter.getIndexAttr(inputType.getRank() ? inputType.getShape()[0] : 1);
  auto insert1 = tensor::InsertSliceOp::create(rewriter, loc, opb, insert0,
                                               offsets, sizes, strides);

  rewriter.replaceOp(op, insert1);
  return success();
}

/// @brief Convert tt.gather to func.call. BiShengIR captures the func
///        with assumed semantics.
/// @param op The `triton::GatherOp` operation to be rewritten.
/// @param adaptor An adaptor for the operation's operands.
/// @param rewriter A pattern rewriter used to modify the IR.
/// @return A `LogicalResult` indicating whether the rewrite was successful.
LogicalResult
GatherConverter::matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value src = adaptor.getSrc();
  Value idx = adaptor.getIndices();
  Value res = op.getResult();
  auto gatherAxis = op.getAxis();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  llvm::SmallString<kFuncNameCap> funcName = gatherFuncNameBase;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName = gatherFuncNameBase;
    funcName += ("_" + std::to_string(uniqueId++));
  }

  auto resTy = res.getType();
  auto libFnType = rewriter.getFunctionType(
      {src.getType(), idx.getType(), rewriter.getI32Type()}, {resTy});
  auto funcOp = func::FuncOp::create(rewriter, loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  Value axis = arith::ConstantIntOp::create(rewriter, loc, gatherAxis, 32);
  auto callOp =
      func::CallOp::create(rewriter, loc, funcOp.getSymNameAttr(),
                           TypeRange({resTy}), ValueRange({src, idx, axis}));

  rewriter.replaceOp(op, callOp);

  return success();
}

LogicalResult
SplitConverter::matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Value input = op.getSrc();
  auto loc = op.getLoc();
  auto inputType = cast<RankedTensorType>(input.getType());

  int64_t rank = inputType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  // Similar to JoinConverter, here adjust last dimension stride
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  strides.back() = rewriter.getIndexAttr(2);

  auto outType = dyn_cast<RankedTensorType>(op.getOutLHS().getType());
  auto sizes = llvm::map_to_vector(outType.getShape(), [&](int64_t t) {
    return OpFoldResult(rewriter.getIndexAttr(t));
  });
  sizes.push_back(rewriter.getIndexAttr(1));

  auto slice0 = tensor::ExtractSliceOp::create(rewriter, loc, outType, input,
                                               offsets, sizes, strides);

  offsets.back() = rewriter.getIndexAttr(1);
  auto slice1 = tensor::ExtractSliceOp::create(rewriter, loc, outType, input,
                                               offsets, sizes, strides);

  SmallVector<Value, 2> slices = {slice0.getResult(), slice1.getResult()};
  rewriter.replaceOp(op, ValueRange(slices));
  return success();
}

/*
the element-wise most significant N bits of the 2N-bit product of x and y
%x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi32>
*/
LogicalResult TritonMulhiuiConverter::matchAndRewrite(
    triton::MulhiUIOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value opl = op.getX();
  Value opr = op.getY();
  Value res = op.getResult();
  auto newMulOp = arith::MulUIExtendedOp::create(rewriter, loc, res.getType(),
                                                 res.getType(), opl, opr);
  // triton only need the high value
  rewriter.replaceOp(op, ValueRange{newMulOp.getHigh()});
  return success();
}

LogicalResult TritonPreciseSqrtConverter::matchAndRewrite(
    triton::PreciseSqrtOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<math::SqrtOp>(op, adaptor.getOperands());
  return success();
}

LogicalResult DevicePrintConverter::matchAndRewrite(
    triton::PrintOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));
  SmallVector<Type, 4> inputTypes;
  for (auto arg : op.getArgs()) {
    inputTypes.push_back(arg.getType());
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp =
      func::FuncOp::create(rewriter, op.getLoc(), printFuncNameBase, libFnType);
  SymbolTable symTab(moduleOp);
  auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
  if (failed(maybePrintFuncNameAttr)) {
    return op->emitError(
        "failed to create a unique func name for device_print");
  }
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  auto prefixAttr = op.getPrefixAttr();
  funcOp->setAttr(prefixAttrName, prefixAttr);
  auto hexAttr = op.getHexAttr();
  funcOp->setAttr(hexAttrName, hexAttr);

  rewriter.setInsertionPoint(op);
  func::CallOp::create(rewriter, op.getLoc(), funcOp, op.getArgs());

  rewriter.eraseOp(op);
  return success();
}

LogicalResult DeviceAssertConverter::matchAndRewrite(
    triton::AssertOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto msgAttr = op.getMessageAttr();
  // Filter out automatically inserted assert ops
  if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(msgAttr)) {
    llvm::StringRef msg = strAttr.getValue();
    if (msg.contains("overflow detected for operation")) {
      rewriter.eraseOp(op);
      return success();
    }
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));
  auto conditionType = op.getCondition().getType();

  auto libFnType = rewriter.getFunctionType({conditionType}, {});
  auto funcOp =
      func::FuncOp::create(rewriter, op.getLoc(), printFuncNameBase, libFnType);
  mlir::SymbolTable symTab(moduleOp);
  auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
  if (failed(maybePrintFuncNameAttr)) {
    return op->emitError(
        "failed to create a unique func name for device_assert");
  }
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  funcOp->setAttr(msgAttrName, msgAttr);

  rewriter.setInsertionPoint(op);
  func::CallOp::create(rewriter, op.getLoc(), funcOp,
                       ValueRange{op.getCondition()});

  rewriter.eraseOp(op);
  return success();
}

LogicalResult
MatmulConverter::matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto opa = adaptor.getA();
  auto opb = adaptor.getB();
  auto opc = adaptor.getC();
  auto dstType = cast<RankedTensorType>(op.getType());
  auto elemTy = dstType.getElementType();
  auto inputPrec = op.getInputPrecision();

  auto createOp = [&](auto &&rewriter, ValueRange operands,
                      ValueRange results) -> Operation * {
    if (dstType.getRank() == 2)
      return linalg::MatmulOp::create(rewriter, op.getLoc(), operands, results);
    else if (dstType.getRank() == 3)
      return linalg::BatchMatmulOp::create(rewriter, op.getLoc(), operands,
                                           results);
    llvm_unreachable("Datatype of DotOp operands could only be 2D or 3D");
  };

  auto replaceOp = [&](auto &&rewriter, ValueRange operands,
                       ValueRange results) -> Operation * {
    if (dstType.getRank() == 2)
      return rewriter.template replaceOpWithNewOp<linalg::MatmulOp>(
          op, operands, results);
    else if (dstType.getRank() == 3)
      return rewriter.template replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, operands, results);
    llvm_unreachable("Datatype of DotOp operands could only be 2D or 3D");
  };

  Operation *matmulOp;
  if (mlir::isa<mlir::FloatType>(elemTy) && !elemTy.isF32()) {
    RankedTensorType opcFp32Ty =
        RankedTensorType::get(dstType.getShape(), rewriter.getF32Type());
    Value opcFp32 =
        arith::ExtFOp::create(rewriter, op.getLoc(), opcFp32Ty, opc);
    matmulOp = createOp(rewriter, ValueRange{opa, opb}, ValueRange{opcFp32});
    auto roundModeAttr = hfusion::RoundModeAttr::get(rewriter.getContext(),
                                                     hfusion::RoundMode::RINT);
    auto truncOp = rewriter.replaceOpWithNewOp<arith::TruncFOp>(
        op, dstType, matmulOp->getResult(0));
    truncOp->setAttr("round_mode", roundModeAttr);
  } else {
    matmulOp = replaceOp(rewriter, ValueRange{opa, opb}, ValueRange{opc});
  }
  matmulOp->setAttr("input_precision",
                    rewriter.getStringAttr(stringifyInputPrecision(inputPrec)));
  return success();
}

LogicalResult
FlipOpConverter::matchAndRewrite(triton::ascend::FlipOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value src = adaptor.getSrc();
  auto rankedSrcTy = cast<RankedTensorType>(src.getType());

  MLIRContext *ctx = rewriter.getContext();

  Type valuesTy = src.getType();
  Location loc = op.getLoc();

  auto dimAttr = op->getAttrOfType<IntegerAttr>("dim");
  if (!dimAttr) {
    op->emitError("missing 'dim' attribute");
    return failure();
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    op->emitError("must be inside a module");
    return failure();
  }

  // Unique callee name: triton_flip, triton_flip_1, …
  std::string funcName = baseFuncName.str();
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName))
    funcName = (baseFuncName + Twine("_") + Twine(uniqueId++)).str();

  auto i64Ty = IntegerType::get(ctx, 64);
  auto libFnType =
      rewriter.getFunctionType({rankedSrcTy, i64Ty}, {rankedSrcTy});

  // Declare the callee
  auto moduleIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto funcOp = func::FuncOp::create(rewriter, loc, funcName, libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  rewriter.restoreInsertionPoint(moduleIP);

  // dim constant
  Value dimVal =
      arith::ConstantIntOp::create(rewriter, loc, dimAttr.getInt(), 64);

  // Call the backend function
  auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
  auto callOp = func::CallOp::create(rewriter, loc, TypeRange({rankedSrcTy}),
                                     callee, ValueRange({src, dimVal}));

  Value finalValues = callOp.getResult(0);

  rewriter.replaceOp(op, {finalValues});
  return success();
}

LogicalResult
SortOpConverter::matchAndRewrite(triton::ascend::SortOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value src = adaptor.getSrc();
  auto rankedSrcTy = cast<RankedTensorType>(src.getType());
  auto srcElemTy = rankedSrcTy.getElementType();
  auto srcShape = rankedSrcTy.getShape();
  auto srcEnc = rankedSrcTy.getEncoding();

  MLIRContext *ctx = rewriter.getContext();

  Type backendElemTy = srcElemTy;
  if (srcElemTy.isInteger(8)) {
    backendElemTy = Float16Type::get(ctx); // i8 -> f16
  } else if (srcElemTy.isInteger(16)) {
    backendElemTy = Float32Type::get(ctx); // i16 -> f32
  }
  Type backendTensorTy = RankedTensorType::get(srcShape, backendElemTy, srcEnc);

  Type valuesTy = src.getType();

  Location loc = op.getLoc();
  auto dimAttr = op->getAttrOfType<IntegerAttr>("dim");
  auto descAttr = op->getAttrOfType<BoolAttr>("descending");
  if (!dimAttr || !descAttr) {
    op->emitError("missing 'dim' or 'descending' attribute");
    return failure();
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    op->emitError("must be inside a module");
    return failure();
  }

  llvm::SmallString<64> baseName("triton_sort");
  llvm::SmallString<64> funcName = baseName;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName = baseName;
    funcName += ("_" + std::to_string(uniqueId++));
  }

  auto i64Ty = IntegerType::get(ctx, 64);
  auto i1Ty = IntegerType::get(ctx, 1);
  auto libFnType = rewriter.getFunctionType({backendTensorTy, i64Ty, i1Ty},
                                            {backendTensorTy});

  auto moduleIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto funcOp = func::FuncOp::create(rewriter, loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  rewriter.restoreInsertionPoint(moduleIP);

  Value srcForCall = src;
  if (backendElemTy != srcElemTy) {
    srcForCall = arith::SIToFPOp::create(rewriter, loc, backendTensorTy, src);
  }

  Value dimVal =
      arith::ConstantIntOp::create(rewriter, loc, dimAttr.getInt(), 64);
  Value descVal = arith::ConstantIntOp::create(rewriter, loc,
                                               descAttr.getValue() ? 1 : 0, 1);

  auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
  auto callOp =
      func::CallOp::create(rewriter, loc, TypeRange({backendTensorTy}), callee,
                           ValueRange({srcForCall, dimVal, descVal}));

  Value valuesFloat = callOp.getResult(0); // tensor<f16/f32>

  Value finalValues = valuesFloat;
  if (backendElemTy != srcElemTy) {
    finalValues = arith::FPToSIOp::create(rewriter, loc, valuesTy, valuesFloat);
  }

  rewriter.replaceOp(op, {finalValues});

  return success();
}

LogicalResult
DotScaledConverter::matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
#if LLVM_VERSION_MAJOR < 22
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
#else
  Value lhs = adaptor.getA();
  Value rhs = adaptor.getB();
#endif
  Value c = adaptor.getC();
#if LLVM_VERSION_MAJOR < 22
  Value lhsScale = adaptor.getLhsScale();
  Value rhsScale = adaptor.getRhsScale();
#else
  Value lhsScale = adaptor.getAScale();
  Value rhsScale = adaptor.getBScale();
#endif
  RankedTensorType dstType = cast<RankedTensorType>(op.getType());

#if LLVM_VERSION_MAJOR >= 22
  auto lhsElemType = op.getAElemType();
  auto rhsElemType = op.getBElemType();

  bool isFP8Input = (lhsElemType == triton::ScaleDotElemType::E4M3 ||
                     lhsElemType == triton::ScaleDotElemType::E5M2) &&
                    (rhsElemType == triton::ScaleDotElemType::E4M3 ||
                     rhsElemType == triton::ScaleDotElemType::E5M2);
  bool isFP4Input = (lhsElemType == triton::ScaleDotElemType::E2M1) &&
                    (rhsElemType == triton::ScaleDotElemType::E2M1);
  if (isFP8Input || isFP4Input) {
    if (!rhsScale) {
      RankedTensorType defaultScaleTy =
          RankedTensorType::get({1}, rewriter.getI8Type());
      Value defaultScaleVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI8IntegerAttr(1));
      Value defaultScaleEmpty =
          tensor::EmptyOp::create(rewriter, loc, defaultScaleTy.getShape(),
                                  defaultScaleTy.getElementType());
      rhsScale =
          linalg::FillOp::create(rewriter, loc, ValueRange{defaultScaleVal},
                                 ValueRange{defaultScaleEmpty})
              .getResult(0);
    }
    Value acc = c ? c
                  : tensor::EmptyOp::create(rewriter, loc, dstType.getShape(),
                                            dstType.getElementType());

    // Get the ScaleDotElemType enum
    // Helper to convert
    auto convertFormat =
        [&](triton::ScaleDotElemType ty) -> mlir::hfusion::DataformatAttr {
      auto ctx = rewriter.getContext();
      switch (ty) {
      case triton::ScaleDotElemType::E2M1:
        return mlir::hfusion::DataformatAttr::get(
            ctx, mlir::hfusion::Dataformat::FP4E2M1_T);
      case triton::ScaleDotElemType::E4M3:
        return mlir::hfusion::DataformatAttr::get(
            ctx, mlir::hfusion::Dataformat::FP8E4M3_T);
      case triton::ScaleDotElemType::E5M2:
        return mlir::hfusion::DataformatAttr::get(
            ctx, mlir::hfusion::Dataformat::FP8E5M2_T);
      default:
        llvm_unreachable("unsupported ScaleDotElemType");
      }
    };

    auto lhsFmt = convertFormat(lhsElemType);
    auto rhsFmt = convertFormat(rhsElemType);

    Value matmulMxResult =
        hfusion::MatMulMxOp::create(rewriter, loc, dstType, lhs, rhs, lhsScale,
                                    rhsScale, acc, lhsFmt, rhsFmt);

    Value finalResult = matmulMxResult;
    if (dstType.getElementType().isBF16()) {
      finalResult =
          arith::TruncFOp::create(rewriter, loc, dstType, matmulMxResult);
    }
    rewriter.replaceOp(op, finalResult);
    return success();
  }
#endif

  if (!lhsScale) {
    return op.emitError("lhsScale is required for non-FP8 input");
  }

  RankedTensorType lhsTy = cast<RankedTensorType>(lhs.getType());
  RankedTensorType lhsScaleTy = cast<RankedTensorType>(lhsScale.getType());
  RankedTensorType rhsScaleTy =
      rhsScale ? cast<RankedTensorType>(rhsScale.getType()) : nullptr;
  RankedTensorType rhsTy = cast<RankedTensorType>(rhs.getType());

  Value lhsScaleOut;
  Value rhsScaleOut;
  Value c127 =
      arith::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI16Type(),
                                rewriter.getI16IntegerAttr(127));
  Value c7 =
      arith::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI16Type(),
                                rewriter.getI16IntegerAttr(7));
  Type i16Ty = rewriter.getI16Type();
  Type bf16Ty = rewriter.getBF16Type();
  Type fp16Ty = rewriter.getF16Type();
  Type fp32Ty = rewriter.getF32Type();
#if LLVM_VERSION_MAJOR >= 22
  bool fastMath = op.getFastMath();
#else
  bool fastMath = false;
#endif

  auto createNanSplat = [&](RankedTensorType tensorTy) -> Value {
    auto floatTy = cast<FloatType>(tensorTy.getElementType());
    auto nanAttr = rewriter.getFloatAttr(
        floatTy, APFloat::getNaN(floatTy.getFloatSemantics()));
    Value empty = tensor::EmptyOp::create(rewriter, loc, tensorTy.getShape(),
                                          tensorTy.getElementType());
    return linalg::FillOp::create(
               rewriter, loc,
               ValueRange{arith::ConstantOp::create(rewriter, loc, nanAttr)},
               ValueRange{empty})
        .getResult(0);
  };

  auto createNaNMask = [&](Value scaleTensor,
                           RankedTensorType scaleTy) -> Value {
    if (scaleTy.getElementType().isIntOrIndex()) {
      auto bitWidth = scaleTy.getElementTypeBitWidth();
      auto allOnes = APInt::getAllOnes(bitWidth);
      auto sentinel = arith::ConstantOp::create(
          rewriter, loc, scaleTy, DenseElementsAttr::get(scaleTy, allOnes));
      return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                   scaleTensor, sentinel)
          .getResult();
    }

    return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::UNO,
                                 scaleTensor, scaleTensor)
        .getResult();
  };

  auto applyNaNMask = [&](Value valueTensor, Value maskTensor) -> Value {
    auto valueTy = cast<RankedTensorType>(valueTensor.getType());
    Value nanTensor = createNanSplat(valueTy);
    return arith::SelectOp::create(rewriter, loc, maskTensor, nanTensor,
                                   valueTensor)
        .getResult();
  };

  if (lhsScaleTy.getElementType().isIntOrIndex()) {
    RankedTensorType lhsScaleI16Ty =
        RankedTensorType::get(lhsScaleTy.getShape(), i16Ty);
    Value lhsScaleI16 =
        arith::ExtSIOp::create(rewriter, op.getLoc(), lhsScaleI16Ty, lhsScale);

    Value lhsShift127Empty = tensor::EmptyOp::create(
        rewriter, op.getLoc(), lhsScaleI16Ty.getShape(), i16Ty);
    Value lhsShift127 =
        linalg::FillOp::create(rewriter, op.getLoc(), ValueRange{c127},
                               ValueRange{lhsShift127Empty})
            .getResult(0);

    Value lhsScaleI16Add127 =
        arith::AddIOp::create(rewriter, op.getLoc(), lhsScaleI16, lhsShift127);

    Value lhsShift7Empty = tensor::EmptyOp::create(
        rewriter, op.getLoc(), lhsScaleI16Ty.getShape(), i16Ty);
    Value lhsShift7 =
        linalg::FillOp::create(rewriter, op.getLoc(), ValueRange{c7},
                               ValueRange{lhsShift7Empty})
            .getResult(0);
    Value lhsScaleI16Shifted = arith::ShLIOp::create(
        rewriter, op.getLoc(), lhsScaleI16Add127, lhsShift7);

    RankedTensorType lhsScaleBF16Ty =
        RankedTensorType::get(lhsScaleTy.getShape(), bf16Ty);
    Value lhsScaleBF16 = arith::BitcastOp::create(
        rewriter, op.getLoc(), lhsScaleBF16Ty, lhsScaleI16Shifted);
    if (lhsTy.getElementType() == fp16Ty) {
      RankedTensorType lhsScaleFp32Ty =
          RankedTensorType::get(lhsScaleTy.getShape(), fp32Ty);
      Value lhsScaleFp32 = arith::ExtFOp::create(rewriter, op.getLoc(),
                                                 lhsScaleFp32Ty, lhsScaleBF16);
      RankedTensorType lhsScaleFp16Ty =
          RankedTensorType::get(lhsScaleTy.getShape(), fp16Ty);
      lhsScaleOut = arith::TruncFOp::create(rewriter, op.getLoc(),
                                            lhsScaleFp16Ty, lhsScaleFp32);
    } else {
      lhsScaleOut = lhsScaleBF16;
    }
  } else {
    lhsScaleOut =
        arith::ExtFOp::create(
            rewriter, op.getLoc(),
            RankedTensorType::get(lhsScaleTy.getShape(), fp32Ty), lhsScale)
            .getResult();
  }

  if (rhsScale && rhsScaleTy.getElementType().isIntOrIndex()) {
    if (rhsScaleTy.getRank() != 2) {
      return op.emitError("rhsScale must be 2D for transpose");
    }

    SmallVector<int64_t> transposedShape = {rhsScaleTy.getShape()[1],
                                            rhsScaleTy.getShape()[0]};
    RankedTensorType transposedRhsScaleTy =
        RankedTensorType::get(transposedShape, rhsScaleTy.getElementType());

    Value transposedRhsScale = triton::TransOp::create(
        rewriter, op.getLoc(), transposedRhsScaleTy, rhsScale,
        DenseI32ArrayAttr::get(rewriter.getContext(), ArrayRef<int32_t>{1, 0}));
    RankedTensorType rhsScaleI16Ty =
        RankedTensorType::get(transposedShape, i16Ty);
    Value rhsScaleI16 = arith::ExtSIOp::create(
        rewriter, op.getLoc(), rhsScaleI16Ty, transposedRhsScale);
    Value rhsShift127Empty = tensor::EmptyOp::create(
        rewriter, op.getLoc(), rhsScaleI16Ty.getShape(), i16Ty);
    Value rhsShift127 =
        linalg::FillOp::create(rewriter, op.getLoc(), ValueRange{c127},
                               ValueRange{rhsShift127Empty})
            .getResult(0);

    Value rhsScaleI16Add127 =
        arith::AddIOp::create(rewriter, op.getLoc(), rhsScaleI16, rhsShift127);
    Value rhsShift7Empty = tensor::EmptyOp::create(
        rewriter, op.getLoc(), rhsScaleI16Ty.getShape(), i16Ty);
    Value rhsShift7 =
        linalg::FillOp::create(rewriter, op.getLoc(), ValueRange{c7},
                               ValueRange{rhsShift7Empty})
            .getResult(0);
    Value rhsScaleI16Shifted = arith::ShLIOp::create(
        rewriter, op.getLoc(), rhsScaleI16Add127, rhsShift7);

    RankedTensorType rhsScaleBF16Ty =
        RankedTensorType::get(transposedShape, bf16Ty);
    Value rhsScaleBF16 = arith::BitcastOp::create(
        rewriter, op.getLoc(), rhsScaleBF16Ty, rhsScaleI16Shifted);

    if (rhsTy.getElementType() == fp16Ty) {
      RankedTensorType rhsScaleFp32Ty =
          RankedTensorType::get(transposedShape, fp32Ty);
      Value rhsScaleFp32 = arith::ExtFOp::create(rewriter, op.getLoc(),
                                                 rhsScaleFp32Ty, rhsScaleBF16);
      RankedTensorType rhsScaleFp16Ty =
          RankedTensorType::get(transposedShape, fp16Ty);
      rhsScaleOut = arith::TruncFOp::create(rewriter, op.getLoc(),
                                            rhsScaleFp16Ty, rhsScaleFp32);
    } else {
      rhsScaleOut = rhsScaleBF16;
    }
    int64_t rhsD0 = rhsScaleTy.getShape()[1];
    int64_t rhsD1 = rhsScaleTy.getShape()[0];
    SmallVector<int64_t> rhsExpandedShape1 = {rhsD0, rhsD1, 1};
    RankedTensorType rhsExpandedTy1 =
        RankedTensorType::get(rhsExpandedShape1, rhsTy.getElementType());
    Value rhsExpanded1 =
        triton::ExpandDimsOp::create(rewriter, op.getLoc(), rhsExpandedTy1,
                                     rhsScaleOut, rewriter.getI32IntegerAttr(2))
            .getResult();

    int64_t rhsDim1 = rhsTy.getShape()[0];
    if (rhsDim1 % rhsD0 != 0) {
      return op.emitError(
          "rhs dim0 must be an integer multiple of rhsScale dim0");
    }
    int64_t rhsD2 = rhsDim1 / rhsD0;
    SmallVector<int64_t> rhsBroadcastShape = {rhsD0, rhsD1, rhsD2};
    RankedTensorType rhsBroadcastTy =
        RankedTensorType::get(rhsBroadcastShape, rhsTy.getElementType());
    Value rhsBroadcasted =
        triton::BroadcastOp::create(rewriter, op.getLoc(), rhsBroadcastTy,
                                    rhsExpanded1)
            .getResult();

    SmallVector<int32_t> transposeOrder = {0, 2, 1};
    Value transposedBroadcasted = triton::TransOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get({rhsD0, rhsD2, rhsD1}, rhsTy.getElementType()),
        rhsBroadcasted,
        DenseI32ArrayAttr::get(rewriter.getContext(), transposeOrder));
    SmallVector<ReassociationIndices> rhsReassociation;
    rhsReassociation.push_back({0, 1});
    rhsReassociation.push_back({2});

    Value scaledRhs = tensor::CollapseShapeOp::create(
                          rewriter, op.getLoc(),
                          RankedTensorType::get({rhsD0 * rhsD2, rhsD1},
                                                rhsTy.getElementType()),
                          transposedBroadcasted, rhsReassociation)
                          .getResult();

    rhs = arith::MulFOp::create(rewriter, op.getLoc(), rhs, scaledRhs)
              .getResult();

    if (!fastMath) {
      Value rhsScaleNaNMask =
          createNaNMask(transposedRhsScale, transposedRhsScaleTy);
      Value rhsExpandedMask =
          triton::ExpandDimsOp::create(
              rewriter, op.getLoc(),
              RankedTensorType::get(rhsExpandedShape1, rewriter.getI1Type()),
              rhsScaleNaNMask, rewriter.getI32IntegerAttr(2))
              .getResult();
      Value rhsBroadcastMask =
          triton::BroadcastOp::create(
              rewriter, op.getLoc(),
              RankedTensorType::get(rhsBroadcastShape, rewriter.getI1Type()),
              rhsExpandedMask)
              .getResult();
      Value transposedBroadcastMask =
          triton::TransOp::create(
              rewriter, op.getLoc(),
              RankedTensorType::get({rhsD0, rhsD2, rhsD1},
                                    rewriter.getI1Type()),
              rhsBroadcastMask,
              DenseI32ArrayAttr::get(rewriter.getContext(), transposeOrder))
              .getResult();
      Value collapsedRhsMask = tensor::CollapseShapeOp::create(
                                   rewriter, op.getLoc(),
                                   RankedTensorType::get({rhsD0 * rhsD2, rhsD1},
                                                         rewriter.getI1Type()),
                                   transposedBroadcastMask, rhsReassociation)
                                   .getResult();
      rhs = applyNaNMask(rhs, collapsedRhsMask);
    }
  }

  int64_t D0 = lhsScaleTy.getShape()[0];
  int64_t D1 = lhsScaleTy.getShape()[1];
  SmallVector<int64_t> expandedShape1 = {D0, D1, 1};
  RankedTensorType expandedTy1 =
      RankedTensorType::get(expandedShape1, lhsTy.getElementType());
  Value expanded1 =
      triton::ExpandDimsOp::create(rewriter, op.getLoc(), expandedTy1,
                                   lhsScaleOut, rewriter.getI32IntegerAttr(2))
          .getResult();

  int64_t lhsDim1 = lhsTy.getShape()[1];
  if (lhsDim1 % D1 != 0) {
    return op.emitError(
        "lhs dim1 must be an integer multiple of lhsScale dim1");
  }
  int64_t D2 = lhsDim1 / D1;
  SmallVector<int64_t> broadcastShape = {D0, D1, D2};
  RankedTensorType broadcastTy =
      RankedTensorType::get(broadcastShape, lhsTy.getElementType());
  Value broadcasted =
      triton::BroadcastOp::create(rewriter, op.getLoc(), broadcastTy, expanded1)
          .getResult();

  SmallVector<ReassociationIndices> reassociation;
  reassociation.push_back({0});
  reassociation.push_back({1, 2});

  Value scaledLhs =
      tensor::CollapseShapeOp::create(
          rewriter, op.getLoc(),
          RankedTensorType::get({D0, D1 * D2}, lhsTy.getElementType()),
          broadcasted, reassociation)
          .getResult();

  Value scaledLhsFinal =
      arith::MulFOp::create(rewriter, op.getLoc(), lhs, scaledLhs).getResult();

  if (!fastMath) {
    Value lhsScaleNaNMask = createNaNMask(lhsScale, lhsScaleTy);
    Value lhsExpandedMask =
        triton::ExpandDimsOp::create(
            rewriter, op.getLoc(),
            RankedTensorType::get(expandedShape1, rewriter.getI1Type()),
            lhsScaleNaNMask, rewriter.getI32IntegerAttr(2))
            .getResult();
    Value lhsBroadcastMask =
        triton::BroadcastOp::create(
            rewriter, op.getLoc(),
            RankedTensorType::get(broadcastShape, rewriter.getI1Type()),
            lhsExpandedMask)
            .getResult();
    Value collapsedLhsMask =
        tensor::CollapseShapeOp::create(
            rewriter, op.getLoc(),
            RankedTensorType::get({D0, D1 * D2}, rewriter.getI1Type()),
            lhsBroadcastMask, reassociation)
            .getResult();
    scaledLhsFinal = applyNaNMask(scaledLhsFinal, collapsedLhsMask);
  }

  Operation *matmulOp;
  if (dstType.getRank() == 2) {
    matmulOp = linalg::MatmulOp::create(
        rewriter, op.getLoc(), ValueRange{scaledLhsFinal, rhs}, ValueRange{c});
  } else if (dstType.getRank() == 3) {
    matmulOp = linalg::BatchMatmulOp::create(
        rewriter, op.getLoc(), ValueRange{scaledLhsFinal, rhs}, ValueRange{c});
  } else {
    return op.emitError("DotScaledOp only support 2D or 3D tensor");
  }

  rewriter.replaceOp(op, matmulOp->getResults());
  return success();
}

// A select whose result is a scalar pointer has no single structured memref
// base. It is lowered through a complete integer address instead: materialize
// both candidate addresses, select on i64, then rebuild the scalar pointer
// carrier with hivm.pointer_cast.
static bool isScalarPointerSelect(arith::SelectOp op) {
  auto pointerType = dyn_cast<triton::PointerType>(op.getType());
  return pointerType && !isa<ShapedType>(pointerType.getPointeeType());
}

// Converts a memref descriptor into the complete byte address represented by
// that descriptor. extract_aligned_pointer_as_index yields the aligned buffer
// pointer; the descriptor's element offset must therefore be converted to
// bytes and added explicitly.
static FailureOr<Value>
materializePointerAddress(Value value, Location loc,
                          ConversionPatternRewriter &rewriter) {
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return failure();
  Type elementType = memrefType.getElementType();
  if (!elementType.isIntOrFloat())
    return failure();

  // A reinterpret_cast exposes its dynamic element offset as a plain operand,
  // and its source carries the un-offset base buffer. Prefer both over
  // extract_strided_metadata, which downstream bishengir passes reject when
  // its root is a function-argument-backed cast.
  if (auto castOp = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    if (castOp.getMixedOffsets().size() != 1)
      return failure();
    Value address = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, castOp.getSource());
    OpFoldResult elementOffset = castOp.getMixedOffsets().front();
    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;
    Value offsetVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, elementOffset);
    if (elementBytes != 1) {
      Value scale = arith::ConstantIndexOp::create(rewriter, loc, elementBytes);
      offsetVal = arith::MulIOp::create(rewriter, loc, offsetVal, scale);
    }
    address = arith::AddIOp::create(rewriter, loc, address, offsetVal);
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                      address)
        .getResult();
  }

  Value address =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, value);
  int64_t staticOffset = memrefType.getStridesAndOffset().second;
  if (staticOffset != 0) {
    Value elementOffset;
    if (ShapedType::isDynamic(staticOffset)) {
      elementOffset =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, value)
              .getOffset();
    } else {
      elementOffset =
          arith::ConstantIndexOp::create(rewriter, loc, staticOffset);
    }

    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;
    if (elementBytes != 1) {
      Value scale = arith::ConstantIndexOp::create(rewriter, loc, elementBytes);
      elementOffset =
          arith::MulIOp::create(rewriter, loc, elementOffset, scale);
    }
    address = arith::AddIOp::create(rewriter, loc, address, elementOffset);
  }
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                    address)
      .getResult();
}

LogicalResult PointerSelectConverter::matchAndRewrite(
    arith::SelectOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!isScalarPointerSelect(op))
    return failure();

  // Mirror the TritonTypeConverter rule for scalar pointers
  // (tt.ptr<T> -> memref<?xT>, with ptr<i1> normalized to memref<?xi8>).
  auto pointerType = cast<triton::PointerType>(op.getType());
  Type pointeeType = pointerType.getPointeeType();
  if (auto intType = dyn_cast<IntegerType>(pointeeType);
      intType && intType.getWidth() == 1) {
    pointeeType = IntegerType::get(op.getContext(), 8);
  }
  auto carrierType = MemRefType::get({ShapedType::kDynamic}, pointeeType);

  FailureOr<Value> trueAddress =
      materializePointerAddress(adaptor.getTrueValue(), op.getLoc(), rewriter);
  FailureOr<Value> falseAddress =
      materializePointerAddress(adaptor.getFalseValue(), op.getLoc(), rewriter);
  if (failed(trueAddress) || failed(falseAddress))
    return rewriter.notifyMatchFailure(
        op, "selected pointer values have no complete integer address");

  auto selectedAddress = arith::SelectOp::create(
      rewriter, op.getLoc(), rewriter.getI64Type(), adaptor.getCondition(),
      *trueAddress, *falseAddress);
  selectedAddress->setAttrs(op->getAttrs());

  auto pointerCast =
      hivm::PointerCastOp::create(rewriter, op.getLoc(), carrierType,
                                  ValueRange{selectedAddress.getResult()});
  rewriter.replaceOp(op, pointerCast.getResult());
  return success();
}
// ==== end port from triton-ascend ====

LogicalResult
PtrToIntConverter::matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value ptr = adaptor.getSrc();

  if (!mlir::isa<MemRefType>(ptr.getType())) {
    return rewriter.notifyMatchFailure(op, "input is not a memref type");
  }

  auto resultType = op.getType();

  // memref.extract_aligned_pointer_as_index is used to obtain the integer
  // representation of the base address.
  auto ptrToIndexOp =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, ptr);

  Value intResult =
      arith::IndexCastOp::create(rewriter, loc, resultType, ptrToIndexOp);

  rewriter.replaceOp(op, intResult);
  return success();
}

LogicalResult
IndexPutConverter::matchAndRewrite(triton::ascend::IndexPutOp op,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto ptr = adaptor.getPtr();
  auto index = op.getIndex();
  auto value = op.getValue();
  auto dim = op.getDim();
  auto indexBoundary = op.getIndexBoundary();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();
  auto dstStride = adaptor.getDstStride();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto ptrTy = dyn_cast<MemRefType>(ptr.getType());
  if (!ptrTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for ptr");
  }
  SmallVector<Type> inputTypes({ptrTy, index.getType(), value.getType(),
                                dim.getType(), indexBoundary.getType()});
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());
  inputTypes.append(dstStride.getTypes().begin(), dstStride.getTypes().end());
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = func::FuncOp::create(rewriter, loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({ptr, index, value, dim, indexBoundary});
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  inputVals.append(dstStride.begin(), dstStride.end());
  func::CallOp::create(rewriter, loc, funcOp.getSymNameAttr(), TypeRange({}),
                       inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult GatherOutToUbConverter::matchAndRewrite(
    triton::ascend::GatherOutToUbOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  auto index = op.getIndex();
  auto indexBoundary = op.getIndexBoundary();
  auto dim = op.getDim();
  auto srcStride = op.getSrcStride();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();
  auto other = op.getOther();

  auto res = op.getResult();
  auto resTy = res.getType();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }

  SmallVector<Type> inputTypes(
      {srcTy, index.getType(), indexBoundary.getType(), dim.getType()});
  inputTypes.append(srcStride.getTypes().begin(), srcStride.getTypes().end());
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());
  if (other)
    inputTypes.push_back(other.getType());

  auto libFnType = rewriter.getFunctionType(inputTypes, {resTy});
  auto funcOp = func::FuncOp::create(rewriter, loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, index, indexBoundary, dim});
  inputVals.append(srcStride.begin(), srcStride.end());
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  if (other)
    inputVals.push_back(other);
  auto callOp = func::CallOp::create(rewriter, loc, funcOp.getSymNameAttr(),
                                     TypeRange({resTy}), inputVals);
  rewriter.replaceOp(op, callOp);
  return success();
}

LogicalResult ScatterUbToOutConverter::matchAndRewrite(
    triton::ascend::ScatterUbToOutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto ptr = adaptor.getPtr();
  auto value = op.getValue();
  auto index = op.getIndex();
  auto indexBoundary = op.getIndexBoundary();
  auto dim = op.getDim();
  auto dstStride = op.getDstStride();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto ptrTy = dyn_cast<MemRefType>(ptr.getType());
  if (!ptrTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for ptr");
  }

  SmallVector<Type> inputTypes({ptrTy, value.getType(), index.getType(),
                                indexBoundary.getType(), dim.getType()});
  inputTypes.append(dstStride.getTypes().begin(), dstStride.getTypes().end());
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());

  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = func::FuncOp::create(rewriter, loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({ptr, value, index, indexBoundary, dim});
  inputVals.append(dstStride.begin(), dstStride.end());
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  func::CallOp::create(rewriter, loc, funcOp.getSymNameAttr(), TypeRange({}),
                       inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult IndirectLoadConverter::matchAndRewrite(
    triton::ascend::IndirectLoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  auto offsets = op.getOffsets();
  auto mask = op.getMask();
  auto other = op.getOther();
  auto res = op.getResult();
  auto resTy = res.getType();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }
  SmallVector<Type> inputTypes({srcTy, offsets.getType()});
  if (mask)
    inputTypes.push_back(mask.getType());
  if (other)
    inputTypes.push_back(other.getType());
  auto libFnType = rewriter.getFunctionType(inputTypes, {resTy});
  auto funcOp = func::FuncOp::create(rewriter, loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, offsets});
  if (mask)
    inputVals.push_back(mask);
  if (other)
    inputVals.push_back(other);
  auto callOp = func::CallOp::create(rewriter, loc, funcOp.getSymNameAttr(),
                                     TypeRange({resTy}), inputVals);
  rewriter.replaceOp(op, callOp);
  return success();
}

LogicalResult IndirectStoreConverter::matchAndRewrite(
    triton::ascend::IndirectStoreOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  auto offsets = op.getOffsets();
  auto value = op.getValue();
  auto mask = op.getMask();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }
  SmallVector<Type> inputTypes({srcTy, offsets.getType(), value.getType()});
  if (mask)
    inputTypes.push_back(mask.getType());

  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = func::FuncOp::create(rewriter, loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, offsets, value});
  if (mask)
    inputVals.push_back(mask);
  func::CallOp::create(rewriter, loc, funcOp.getSymNameAttr(), TypeRange({}),
                       inputVals);
  rewriter.eraseOp(op);
  return success();
}

IndexSelectSimdConverter::IndexSelectSimdConverter(MLIRContext *context)
    : OpConversionPattern<triton::ascend::IndexSelectSimdOp>(context) {}

LogicalResult IndexSelectSimdConverter::matchAndRewrite(
    triton::ascend::IndexSelectSimdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // Get converted operands
  Value src = adaptor.getSrc();
  Value indexTensor = adaptor.getIndex();
  auto srcShapeVals = adaptor.getSrcShape();
  auto srcOffsetVals = adaptor.getSrcOffset();
  auto readShapeAttr = op.getReadShape();
  int32_t dim = op.getDim();

  // Get result type
  auto resultTensorType = cast<RankedTensorType>(op.getResult().getType());
  auto elemType = resultTensorType.getElementType();
  auto resultShape = resultTensorType.getShape();

  // Convert src (tt.ptr -> memref) to the correct memref shape
  // src is now memref<?xT> after type conversion, need to reinterpret to full
  // shape
  auto srcMemRefType = cast<MemRefType>(src.getType());

  // DenseI32ArrayAttr can be implicitly converted to ArrayRef<int32_t>
  ArrayRef<int32_t> readShape = readShapeAttr;

  // Helper lambda to convert Value to Index type if needed
  auto toIndexValue = [&](Value val) -> Value {
    if (!val.getType().isIndex()) {
      return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                        val);
    }
    return val;
  };

  // Build multi-dimensional memref type and determine static sizes
  // Merge two passes: build fullSrcShape and staticSizes in one loop
  SmallVector<int64_t> fullSrcShape;
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> sizes;

  for (size_t i = 0; i < srcShapeVals.size(); ++i) {
    bool isDynamicDim = (i == static_cast<size_t>(dim) && readShape[i] == -1);
    int64_t staticSize;

    if (isDynamicDim) {
      // Dynamic dimension: readShape[i] == -1 indicates dynamic
      staticSize = ShapedType::kDynamic;
      fullSrcShape.push_back(ShapedType::kDynamic);
      sizes.push_back(toIndexValue(srcShapeVals[i]));
    } else if (auto constOp =
                   srcShapeVals[i].getDefiningOp<arith::ConstantIndexOp>()) {
      // Static dimension: use constant value
      staticSize = constOp.value();
      fullSrcShape.push_back(staticSize);
    } else {
      // Runtime value: must use dynamic
      staticSize = ShapedType::kDynamic;
      fullSrcShape.push_back(ShapedType::kDynamic);
      sizes.push_back(toIndexValue(srcShapeVals[i]));
    }
    staticSizes.push_back(staticSize);
  }
  auto fullSrcMemRefType = MemRefType::get(fullSrcShape, elemType);

  // Build static offsets, sizes, and strides for ReinterpretCastOp
  SmallVector<Value> offsets, strides;
  SmallVector<int64_t> staticOffsets, staticStrides;
  staticOffsets.push_back(0); // offsets are 0

  // Calculate static strides: stride[i] = product of all dimensions after i
  for (size_t i = 0; i < srcShapeVals.size(); ++i) {
    int64_t staticStride = 1;
    bool isDynamic = false;

    // Check if stride needs to be dynamic (any dimension after i is dynamic)
    for (size_t j = i + 1; j < srcShapeVals.size(); ++j) {
      if (staticSizes[j] == ShapedType::kDynamic) {
        isDynamic = true;
        break;
      }
      staticStride *= staticSizes[j];
    }

    if (isDynamic) {
      staticStride = ShapedType::kDynamic;
      // Compute stride dynamically: stride[i] = product of sizes after i
      Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, 1);
      for (size_t j = i + 1; j < srcShapeVals.size(); ++j) {
        if (staticSizes[j] != ShapedType::kDynamic) {
          strideVal = arith::MulIOp::create(
              rewriter, loc, strideVal,
              arith::ConstantIndexOp::create(rewriter, loc, staticSizes[j]));
        } else {
          // Dynamic dimension: use runtime value
          strideVal = arith::MulIOp::create(rewriter, loc, strideVal,
                                            toIndexValue(srcShapeVals[j]));
        }
      }
      strides.push_back(strideVal);
    }
    staticStrides.push_back(staticStride);
  }

  auto srcMemRef = memref::ReinterpretCastOp::create(
      rewriter, loc, fullSrcMemRefType, src, offsets, sizes, strides,
      staticOffsets, staticSizes, staticStrides);

  // Allocate output buffer
  auto resultMemRefType = MemRefType::get(resultShape, elemType);
  auto outputBuffer = memref::AllocOp::create(rewriter, loc, resultMemRefType);

  // Get indices tensor type for extracting
  auto indicesTensorType = cast<RankedTensorType>(indexTensor.getType());
  int64_t numIndices = indicesTensorType.getShape()[0];

  // Create for loop
  auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto numIndicesVal =
      arith::ConstantIndexOp::create(rewriter, loc, numIndices);
  auto stepOne = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto forOp =
      scf::ForOp::create(rewriter, loc, zeroIdx, numIndicesVal, stepOne);

  // Mark as parallel loop
  forOp->setAttr("hivm.parallel_loop", rewriter.getUnitAttr());

  // Build loop body
  Block *loopBody = forOp.getBody();
  auto savedInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(loopBody);

  // Remove the terminator temporarily
  Operation *terminator = &loopBody->back();
  rewriter.setInsertionPoint(terminator);

  Value iv = forOp.getInductionVar();

  // Extract index from indices tensor
  Value selectedIdx =
      tensor::ExtractOp::create(rewriter, loc, indexTensor, ValueRange{iv});
  Value selectedIdxAsIndex = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getIndexType(), selectedIdx);

  // Build source subview offsets/sizes/strides
  SmallVector<OpFoldResult> srcSubviewOffsets, srcSubviewSizes,
      srcSubviewStrides;

  for (size_t i = 0; i < srcOffsetVals.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      // Use the selected index for this dimension
      srcSubviewOffsets.push_back(selectedIdxAsIndex);
      srcSubviewSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      // Use provided offset and read size for other dimensions
      Value offsetVal = srcOffsetVals[i];
      if (!offsetVal.getType().isIndex()) {
        offsetVal = arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), offsetVal);
      }
      srcSubviewOffsets.push_back(offsetVal);
      srcSubviewSizes.push_back(rewriter.getIndexAttr(readShape[i]));
    }
    srcSubviewStrides.push_back(rewriter.getIndexAttr(1));
  }

  auto srcSubview =
      memref::SubViewOp::create(rewriter, loc, srcMemRef, srcSubviewOffsets,
                                srcSubviewSizes, srcSubviewStrides);

  // Build destination subview
  SmallVector<OpFoldResult> dstSubviewOffsets, dstSubviewSizes,
      dstSubviewStrides;
  for (size_t i = 0; i < resultShape.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      dstSubviewOffsets.push_back(iv);
      dstSubviewSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      dstSubviewOffsets.push_back(rewriter.getIndexAttr(0));
      dstSubviewSizes.push_back(rewriter.getIndexAttr(readShape[i]));
    }
    dstSubviewStrides.push_back(rewriter.getIndexAttr(1));
  }

  auto dstSubview =
      memref::SubViewOp::create(rewriter, loc, outputBuffer, dstSubviewOffsets,
                                dstSubviewSizes, dstSubviewStrides);

  // Check if index_select is on the trailing axis (last dimension)
  if (static_cast<size_t>(dim) == fullSrcShape.size() - 1) {
    // For index_select on the trailing axis, mark as discrete memory access
    // This degrades to scalar read/write handling to avoid alignment issues
    auto copyOp = memref::CopyOp::create(rewriter, loc, srcSubview, dstSubview);
    copyOp->setAttr(ConverterUtils::discreteAttrName, rewriter.getUnitAttr());
  } else {
    // For index_select on non-trailing axes, add stride alignment annotation
    // This tells the backend to handle address alignment for DMA operations
    auto dstMarkOp = annotation::MarkOp::create(rewriter, loc, dstSubview);
    dstMarkOp->setAttr(
        "hfusion.stride_align_dims",
        rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(dim)}));
    dstMarkOp->setAttr("hfusion.stride_align_value_in_byte",
                       rewriter.getDenseI32ArrayAttr({32}));

    // Copy from source to destination
    memref::CopyOp::create(rewriter, loc, srcSubview, dstSubview);
  }

  // Restore insertion point
  rewriter.restoreInsertionPoint(savedInsertionPoint);

  // Convert memref to tensor
  auto resultTensor = bufferization::ToTensorOp::create(
      rewriter, loc, resultTensorType, outputBuffer, true, true);

  // Mark as index_select_simd
  resultTensor->setAttr("index_select_simd", rewriter.getUnitAttr());

  // Replace the original op
  rewriter.replaceOp(op, resultTensor);

  return success();
}

} // namespace TTOpConverters
