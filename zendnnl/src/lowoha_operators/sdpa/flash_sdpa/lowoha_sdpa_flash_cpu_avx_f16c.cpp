/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

/*
 * AVX + F16C + FMA3 tier of the flash SDPA kernels: 8 float lanes with a
 * fused multiply-add and hardware FP16 conversion.
 *
 * Within AMD family 15h this covers Piledriver and Steamroller (Excavator
 * uses the AVX2 tier instead). Bulldozer must not reach it: it has neither
 * FMA3 nor F16C, which is why select_simd_tier() requires both.
 *
 * See lowoha_sdpa_flash_cpu.cpp for why the tier has to be selected per
 * translation unit rather than at the call site.
 */

#define ZENDNNL_SDPA_FLASH_TIER_AVX_F16C 1
#include "lowoha_sdpa_flash_cpu.cpp"
