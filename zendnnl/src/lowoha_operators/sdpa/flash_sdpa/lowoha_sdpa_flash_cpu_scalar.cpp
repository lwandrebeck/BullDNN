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
 * Scalar tier of the flash SDPA kernels.
 *
 * This builds the flash SDPA body a second time with no AVX-512 target
 * pragma in effect, and contributes only sdpa_flash_run_scalar_internal();
 * the primary translation unit keeps the public entry points and the
 * AVX-512 instantiations.
 *
 * Why a separate translation unit is necessary: GCC applies a
 * `#pragma GCC target(...)` region to every function *defined* inside it,
 * template instantiations included. In the primary unit that region enables
 * `fma`, so the scalar_tag instantiations came out containing FMA3
 * (vfmadd*ss) even though they touch no vector type — and FMA3 faults on
 * AMD Bulldozer, the one family-15h core that lacks it. Moving the
 * instantiation to a unit compiled without that pragma is what keeps the
 * scalar fallback executable on every supported CPU. Selecting the tier per
 * translation unit is also the only thing that works: the options in effect
 * at the *call* site have no bearing on how an instantiation is compiled.
 *
 * Including the implementation file is deliberate, so the kernel bodies stay
 * in one place rather than being duplicated or moved into a header.
 */

#define ZENDNNL_SDPA_FLASH_TIER_SCALAR 1
#include "lowoha_sdpa_flash_cpu.cpp"
