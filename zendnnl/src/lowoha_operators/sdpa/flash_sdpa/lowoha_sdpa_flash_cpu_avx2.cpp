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
 * AVX2 + F16C + FMA3 tier of the flash SDPA kernels: 8 float lanes, with
 * 256-bit integer conversions for bf16.
 *
 * Within AMD family 15h only Excavator has AVX2. This tier is also what any
 * AVX2 host without AVX-512 runs (Haswell, Zen 1 to Zen 3).
 *
 * See lowoha_sdpa_flash_cpu.cpp for why the tier has to be selected per
 * translation unit rather than at the call site.
 */

#define ZENDNNL_SDPA_FLASH_TIER_AVX2 1
#include "lowoha_sdpa_flash_cpu.cpp"
