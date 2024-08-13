/*
 * Copyright (c) 2017-2024 The Forge Interactive Inc.
 *
 * This file is part of The-Forge
 * (see https://github.com/ConfettiFX/The-Forge).
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Unit Test for testing transformations using a solar system.
// Tests the basic mat4 transformations, such as scaling, rotation, and translation.

#include <cstdint>

// Interfaces
#include "Common_3/Application/Interfaces/IApp.h"
#include "Common_3/Application/Interfaces/ICameraController.h"
#include "Common_3/Application/Interfaces/IFont.h"
#include "Common_3/Application/Interfaces/IInput.h"
#include "Common_3/Application/Interfaces/IProfiler.h"
#include "Common_3/Application/Interfaces/IScreenshot.h"
#include "Common_3/Application/Interfaces/IUI.h"
#include "Forge/TF_FileSystem.h"
#include "Forge/Formats/ply/TF_ply.h"
#include "Forge/TF_Log.h"
#include "Forge/Core/TF_Time.h"

#include "Forge/Math/TF_Types.h"
#include "Forge/Math/TF_Simd32x4.h"

#include "Common_3/Utilities/RingBuffer.h"

// Renderer
#include "Forge/Graphics/TF_Graphics.h"
#include "Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h"

// Math
#include "Forge/Core/TF_Math.h"
#include "Forge/Graphics/TF_GPUConfig.h"

#include "Forge/Mem/TF_Memory.h"
#include "TF/Forge/Math/TF_FastHash.h"

///// Demo structures
//struct PlanetInfoStruct
//{
//    mat4  mTranslationMat;
//    mat4  mScaleMat;
//    mat4  mSharedMat; // Matrix to pass down to children
//    vec4  mColor;
//    uint  mParentIndex;
//    float mYOrbitSpeed; // Rotation speed around parent
//    float mZOrbitSpeed;
//    float mRotationSpeed; // Rotation speed around self
//    float mMorphingSpeed; // Speed of morphing betwee cube and sphere
//};

struct UniformBlock
{
    CameraMatrix mProjectView;
};
uint64_t mNumOfPoints;

//struct UniformBlockSky
//{
//    CameraMatrix mProjectView;
//};

// But we only need Two sets of resources (one in flight and one being used on CPU)
const uint32_t gDataBufferCount = 2;
const uint     gTimeOffset = 600000; // For visually better starting locations
const float    gRotSelfScale = 0.0004f;
const float    gRotOrbitYScale = 0.001f;
const float    gRotOrbitZScale = 0.00001f;

const hash32_t gPycFeaturesReset[] = {
    tfStrHash32(tfCToStrRef("f_rest_0")),  tfStrHash32(tfCToStrRef("f_rest_1")),  tfStrHash32(tfCToStrRef("f_rest_2")),
    tfStrHash32(tfCToStrRef("f_rest_3")),  tfStrHash32(tfCToStrRef("f_rest_4")),  tfStrHash32(tfCToStrRef("f_rest_5")),
    tfStrHash32(tfCToStrRef("f_rest_6")),  tfStrHash32(tfCToStrRef("f_rest_7")),  tfStrHash32(tfCToStrRef("f_rest_8")),
    tfStrHash32(tfCToStrRef("f_rest_9")),  tfStrHash32(tfCToStrRef("f_rest_10")), tfStrHash32(tfCToStrRef("f_rest_11")),
    tfStrHash32(tfCToStrRef("f_rest_12")), tfStrHash32(tfCToStrRef("f_rest_13")), tfStrHash32(tfCToStrRef("f_rest_14")),
    tfStrHash32(tfCToStrRef("f_rest_15")), tfStrHash32(tfCToStrRef("f_rest_16")), tfStrHash32(tfCToStrRef("f_rest_17")),
    tfStrHash32(tfCToStrRef("f_rest_18")), tfStrHash32(tfCToStrRef("f_rest_19")), tfStrHash32(tfCToStrRef("f_rest_20")),
    tfStrHash32(tfCToStrRef("f_rest_21")), tfStrHash32(tfCToStrRef("f_rest_22")), tfStrHash32(tfCToStrRef("f_rest_23")),
    tfStrHash32(tfCToStrRef("f_rest_24")), tfStrHash32(tfCToStrRef("f_rest_25")), tfStrHash32(tfCToStrRef("f_rest_26")),
    tfStrHash32(tfCToStrRef("f_rest_27")), tfStrHash32(tfCToStrRef("f_rest_28")), tfStrHash32(tfCToStrRef("f_rest_29")),
    tfStrHash32(tfCToStrRef("f_rest_30")), tfStrHash32(tfCToStrRef("f_rest_31")), tfStrHash32(tfCToStrRef("f_rest_32")),
    tfStrHash32(tfCToStrRef("f_rest_33")), tfStrHash32(tfCToStrRef("f_rest_34")), tfStrHash32(tfCToStrRef("f_rest_35")),
    tfStrHash32(tfCToStrRef("f_rest_36")), tfStrHash32(tfCToStrRef("f_rest_37")), tfStrHash32(tfCToStrRef("f_rest_38")),
    tfStrHash32(tfCToStrRef("f_rest_39")), tfStrHash32(tfCToStrRef("f_rest_40")), tfStrHash32(tfCToStrRef("f_rest_41")),
    tfStrHash32(tfCToStrRef("f_rest_42")), tfStrHash32(tfCToStrRef("f_rest_43")), tfStrHash32(tfCToStrRef("f_rest_44")),
};
const uint32_t gFeatureReset = TF_ARRAY_COUNT(gPycFeaturesReset) / 3;

RendererContext* pContext = NULL;
Renderer*        pRenderer = NULL;

Queue*     pGraphicsQueue = NULL;
GpuCmdRing gGraphicsCmdRing = {};

SwapChain*    pSwapChain = NULL;
RenderTarget* pDepthBuffer = NULL;
Semaphore*    pImageAcquiredSemaphore = NULL;

Shader* pParticleShader = NULL;
Pipeline* pParticlePipeline = NULL;

RootSignature* pRootSignature = NULL;
Buffer* pGaussianPosition = NULL;
Buffer* pFeatureRest = NULL;
Buffer* pGaussianColor = NULL;
Buffer* pProjViewUniformBuffer[gDataBufferCount] = { NULL };

DescriptorSet* pDescriptorSetUniforms = { NULL };

uint32_t     gFrameIndex = 0;
ProfileToken gGpuProfileToken = PROFILE_INVALID_TOKEN;

UniformBlock     gUniformData;
ICameraController* pCameraController = NULL;

UIComponent* pGuiWindow = NULL;

uint32_t gFontID = 0;
QueryPool* pPipelineStatsQueryPool[gDataBufferCount] = {};

FontDrawDesc gFrameTimeDraw;

Tsimd_f32x4_t* pPointPos;
Tsimd_f32x4_t* pPointColor;


static unsigned char gPipelineStatsCharArray[2048] = {};
static bstring       gPipelineStats = bfromarr(gPipelineStatsCharArray);

void reloadRequest(void*)
{
    ReloadDesc reload{ RELOAD_TYPE_SHADER };
    requestReload(&reload);
}


struct TPlyArgs4x4_s {
    TStrSpan mCol0[4];
    TStrSpan mCol1[4];
    TStrSpan mCol2[4];
    TStrSpan mCol3[4];
};
static inline bool plyUtilReadf32x3(FileStream* stream, struct TPlyReader* reader, size_t cursor, struct TPlyElement* element,
                                 hash32_t args[3], struct Tf32x3_s* value); 
//static inline bool plyUtilReadf32x4x4(FileStream* stream, struct TPlyReader* reader, size_t cursor, struct TPlyElement* element,
//                                struct TPlyArgs4x4_s args, struct Tf32x4x4_s* value);
static inline bool plyUtilReadf32x4(FileStream* stream, struct TPlyReader* reader, size_t cursor, struct TPlyElement* element,
                                 hash32_t args[4], struct Tf32x4_s* value);

//static inline bool plyUtilReadf32x4x4(FileStream* stream, struct TPlyReader* reader, size_t cursor, struct TPlyElement* element,
//                                struct TPlyArgs4x4_s args, struct Tf32x4x4_s* value) {
//    struct TPlyAttribResult findAttrib;
//    struct TPlyNumber       number;
//    for (size_t i = 0; i < 3; i++) {
//        if (!plyUtilReadf32x4(stream, reader, cursor, element, args.mCol0, &value->mCol0))
//            return false;
//        if (!plyUtilReadf32x4(stream, reader, cursor, element, args.mCol0, &value->mCol1))
//            return false;
//        if (!plyUtilReadf32x4(stream, reader, cursor, element, args.mCol0, &value->mCol2))
//            return false;
//        if (!plyUtilReadf32x4(stream, reader, cursor, element, args.mCol0, &value->mCol3))
//            return false;
//    }
//
//    return true;
//}


static inline bool plyUtilReadf32x4(FileStream* stream, struct TPlyReader* reader, size_t cursor, struct TPlyElement* element,
                                 hash32_t args[4], struct Tf32x4_s* value) {
    struct TPlyAttribResult findAttrib;
    struct TPlyNumber       number;
    for (size_t i = 0; i < 4; i++) {
        if (!tfPlyFindAttribRef(stream, reader, cursor, element, args[i], &findAttrib))
            return false;
        if (!tfPlyDecodeNumber(stream, findAttrib.mCursor, reader->mFormat, findAttrib.mType, &number))
            return false;
        value->v[i] = number.flt;
    }

    return true;
}


static inline bool plyUtilReadf32x3(FileStream* stream, struct TPlyReader* reader, size_t cursor, struct TPlyElement* element,
                                 hash32_t args[3], struct Tf32x3_s* value) {
    struct TPlyAttribResult findAttrib;
    struct TPlyNumber       number;
    for (size_t i = 0; i < 3; i++) {
        if (!tfPlyFindAttribRef(stream, reader, cursor, element, args[i], &findAttrib))
            return false;
        if (!tfPlyDecodeNumber(stream, findAttrib.mCursor, reader->mFormat, findAttrib.mType, &number))
            return false;
        value->v[i] = number.flt;
    }
    return true;
}

class Transformations: public IApp
{
public:
    bool Init()
    {
        // FILE PATHS
        fsSetPathForResourceDir(pSystemFileIO, RM_CONTENT, RD_SHADER_BINARIES, "CompiledShaders");
        fsSetPathForResourceDir(pSystemFileIO, RM_CONTENT, RD_TEXTURES, "Textures");
        fsSetPathForResourceDir(pSystemFileIO, RM_CONTENT, RD_FONTS, "Fonts");
        fsSetPathForResourceDir(pSystemFileIO, RM_DEBUG, RD_SCREENSHOTS, "Screenshots");
        fsSetPathForResourceDir(pSystemFileIO, RM_CONTENT, RD_SCRIPTS, "Scripts");
        fsSetPathForResourceDir(pSystemFileIO, RM_DEBUG, RD_DEBUG, "Debug");
        fsSetPathForResourceDir(pSystemFileIO, RM_CONTENT, RD_GPU_CONFIG, "GPUCfg");
        fsSetPathForResourceDir(pSystemFileIO, RM_CONTENT, RD_OTHER_FILES, "Other");

        // window and renderer setup
        RendererContextDesc rendererContextDesc = {};
        memset(&rendererContextDesc, 0, sizeof(RendererContextDesc));
        rendererContextDesc.mApi = (RendererApi)mSettings.mSelectedAPI;
        initRendererContext(GetName(), &rendererContextDesc, &pContext);

        struct GPUConfiguration def = { 0 };
        tfInitGPUConfiguration(&def);
        tfBoostrapDefaultGPUConfiguration(&def);
        GPUConfigSelection selection = tfApplyGPUConfig(&def, pContext);
        RendererDesc settings;
        memset(&settings, 0, sizeof(RendererDesc));
        settings.pContext = pContext;
        settings.pSelectedDevice = selection.mDeviceAdapter;
        settings.mProperties = selection.mGpuProperty;
        settings.mProperties.mPipelineStatsQueries = false;
        initRenderer(GetName(), &settings, &pRenderer);

        tfFreeGPUConfiguration(&def);
        // check for init success
        if (!pRenderer)
            return false;

        if (pRenderer->pProperties->mPipelineStatsQueries)
        {
            QueryPoolDesc poolDesc = {};
            poolDesc.mQueryCount = 3; // The count is 3 due to quest & multi-view use otherwise 2 is enough as we use 2 queries.
            poolDesc.mType = QUERY_TYPE_PIPELINE_STATISTICS;
            for (uint32_t i = 0; i < gDataBufferCount; ++i)
            {
                addQueryPool(pRenderer, &poolDesc, &pPipelineStatsQueryPool[i]);
            }
        }

        QueueDesc queueDesc = {};
        queueDesc.mType = QUEUE_TYPE_GRAPHICS;
        queueDesc.mFlag = QUEUE_FLAG_INIT_MICROPROFILE;
        addQueue(pRenderer, &queueDesc, &pGraphicsQueue);

        GpuCmdRingDesc cmdRingDesc = {};
        cmdRingDesc.pQueue = pGraphicsQueue;
        cmdRingDesc.mPoolCount = gDataBufferCount;
        cmdRingDesc.mCmdPerPoolCount = 1;
        cmdRingDesc.mAddSyncPrimitives = true;
        addGpuCmdRing(pRenderer, &cmdRingDesc, &gGraphicsCmdRing);

        addSemaphore(pRenderer, &pImageAcquiredSemaphore);

        initResourceLoaderInterface(pRenderer);

        {
            FileStream fh = {};
          
            //element vertex 1734607
            //property float x
            //property float y
            //property float z
            //property float nx
            //property float ny
            //property float nz
            //property float f_dc_0
            //property float f_dc_1
            //property float f_dc_2
            //property float f_rest_0
            //property float f_rest_1
            //property float f_rest_2
            //property float f_rest_3
            //property float f_rest_4
            //property float f_rest_5
            //property float f_rest_6
            //property float f_rest_7
            //property float f_rest_8
            //property float f_rest_9
            //property float f_rest_10
            //property float f_rest_11
            //property float f_rest_12
            //property float f_rest_13
            //property float f_rest_14
            //property float f_rest_15
            //property float f_rest_16
            //property float f_rest_17
            //property float f_rest_18
            //property float f_rest_19
            //property float f_rest_20
            //property float f_rest_21
            //property float f_rest_22
            //property float f_rest_23
            //property float f_rest_24
            //property float f_rest_25
            //property float f_rest_26
            //property float f_rest_27
            //property float f_rest_28
            //property float f_rest_29
            //property float f_rest_30
            //property float f_rest_31
            //property float f_rest_32
            //property float f_rest_33
            //property float f_rest_34
            //property float f_rest_35
            //property float f_rest_36
            //property float f_rest_37
            //property float f_rest_38
            //property float f_rest_39
            //property float f_rest_40
            //property float f_rest_41
            //property float f_rest_42
            //property float f_rest_43
            //property float f_rest_44
            //property float opacity
            //property float scale_0
            //property float scale_1
            //property float scale_2
            //property float rot_0
            //property float rot_1
            //property float rot_2
            //property float rot_3

            if (!fsOpenStreamFromPath(RD_OTHER_FILES, "drjohnson/point_cloud/iteration_7000/point_cloud.ply", FM_READ, &fh)) 
                return false;
            struct TPlyReader reader = {};
            if(!tfAddPlyFileReader(&fh, &reader)) {
                LOGF(eERROR, "Failed to load ply.");
                return false;
            }

            size_t cursor = 0;
            struct TPlyElement* element;
            if(!tfPlySeekElementStream(&fh, &reader, tfCToStrRef("vertex"), &element, &cursor)) {
                LOGF(eERROR, "Failed to find vertex stream.");
                return false;
            }

            {
                BufferLoadDesc positionVbDesc = {};
                positionVbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
                positionVbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
                positionVbDesc.mDesc.mSize = sizeof(struct Tf32x3_s) * element->mNumElements;
                positionVbDesc.ppBuffer = &pGaussianPosition;
                addResource(&positionVbDesc, NULL);
            }
            {
                BufferLoadDesc positionShDesc = {};
                positionShDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
                positionShDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
                positionShDesc.mDesc.mSize = sizeof(struct Tf32x3_s) * gFeatureReset * element->mNumElements;
                positionShDesc.ppBuffer = &pFeatureRest;
                addResource(&positionShDesc, NULL);
            }
            {
                BufferLoadDesc colorVbDesc = {};
                colorVbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
                colorVbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
                colorVbDesc.mDesc.mSize = sizeof(struct Tf32x3_s) * element->mNumElements;
                colorVbDesc.ppBuffer = &pGaussianColor;
                addResource(&colorVbDesc, NULL);
            }

            BufferUpdateDesc positionUpdateDesc = { pGaussianPosition };
            BufferUpdateDesc colorUpdateDesc = { pGaussianColor };
            BufferUpdateDesc featureUpdateDesc = { pFeatureRest };
            beginUpdateResource(&positionUpdateDesc);
            beginUpdateResource(&colorUpdateDesc);
            beginUpdateResource(&featureUpdateDesc);
            struct TPlyAttribResult findAttrib;
            mNumOfPoints = element->mNumElements;
            for (size_t eleIdx = 0; eleIdx < element->mNumElements; eleIdx++, cursor += tfPlyNextElement(&fh, &reader, cursor, element)) {
                if(eleIdx % 1000 == 0)
                LOGF(eINFO, "processing element %lu/%lu", eleIdx, element->mNumElements);
                struct Tf32x3_s pos;
                struct TPlyNumber number;
                hash32_t posArgs[3] = {tfStrHash32(tfCToStrRef("x")), tfStrHash32(tfCToStrRef("y")), tfStrHash32(tfCToStrRef("z"))};
                if (!plyUtilReadf32x3(&fh, &reader, cursor, element, posArgs, &pos))
                    return false;

                for (size_t fIdx = 0; fIdx < gFeatureReset; fIdx++) {
                    struct Tf32x3_s feature;
                    hash32_t resetArg[3] = {gPycFeaturesReset[(fIdx * 3)], gPycFeaturesReset[(fIdx * 3) + 1], gPycFeaturesReset[(fIdx * 3) + 2]};
                    if (!plyUtilReadf32x3(&fh, &reader, cursor, element, resetArg, &feature))
                        return false;
                    ((Tf32x3_s*)featureUpdateDesc.pMappedData)[(eleIdx * gFeatureReset) + fIdx] = feature;
                }

                ((Tf32x3_s*)positionUpdateDesc.pMappedData)[eleIdx] = pos;
                ((Tf32x3_s*)colorUpdateDesc.pMappedData)[eleIdx] = pos;
            }
            endUpdateResource(&colorUpdateDesc);
            endUpdateResource(&positionUpdateDesc);
            endUpdateResource(&featureUpdateDesc);

            tfFreePlyFileReader(&reader);
           // gGaussianPoints = (struct GaussianPoint*)tf_malloc(sizeof(GaussianPoint) * mNumOfPoints);
           // pPointPos = (Tsimd_f32x4_t*)tf_malloc(sizeof(Tsimd_f32x4_t) * mNumOfPoints);
           // for(size_t pIdx = 0; pIdx < mNumOfPoints; pIdx++) {
           //     //GaussianPoint point;
           //     GaussianPoint* point = &gGaussianPoints[pIdx];

           //     uint64_t pointId;
           //     Tf64x3_s pos;
           //     Tu8x3_s  color;
           //     double   error;
           //     uint64_t trackLength;

           //     if (fsReadFromStream(&fh, &pointId, sizeof(pointId)) != sizeof(pointId)) return false;
           //     if (fsReadFromStream(&fh, &pos, sizeof(pos)) != sizeof(pos)) return false;
           //     if (fsReadFromStream(&fh, &color, sizeof(color)) != sizeof(color)) return false;
           //     if (fsReadFromStream(&fh, &error, sizeof(error)) != sizeof(error)) return false;
           //     if (fsReadFromStream(&fh, &trackLength, sizeof(trackLength)) != sizeof(trackLength)) return false;
           //     point->mPointId = pointId;
           //     point->mPos = { (float)pos.x, (float)pos.y, (float)pos.z };
           //     point->mColor = { ((float)color.x / 255.0f), ((float)color.y / 255.0f), ((float)color.z / 255.0f) };
           //     point->mTrackLength = trackLength;
           //     point->mError = error;

           //     pPointPos[pIdx] = tfSimdLoad_f32x4(pos.x, pos.y, pos.z, 0.0f);
           //     gGaussianPoints[pIdx].mPointIds = (int*)tf_malloc(sizeof(int) * trackLength);
           //     gGaussianPoints[pIdx].mImageIds = (int*)tf_malloc(sizeof(int) * trackLength);
           //     for (size_t tIdx = 0; tIdx < trackLength; tIdx++) {
           //         int image_idx;
           //         int point2d_idx;
           //         if (fsReadFromStream(&fh, &image_idx, sizeof(image_idx)) != sizeof(image_idx))
           //             return false;
           //         if (fsReadFromStream(&fh, &point2d_idx, sizeof(point2d_idx)) != sizeof(point2d_idx))
           //             return false;
           //         gGaussianPoints[pIdx].mPointIds[tIdx] = point2d_idx;
           //         gGaussianPoints[pIdx].mImageIds[tIdx] = image_idx;
           //     }
           // }
        }

       // // calculate scale
       // {
       //     Tsimd_f32x4_t globalMin = tfSimdLoad_f32x4(0,0,0,0);
       //     Tsimd_f32x4_t globalMax = tfSimdLoad_f32x4(0,0,0,0);

       //     for (size_t i = 0; i < mNumOfPoints; i++) {
       //         globalMax = tfSimdMaxPerElem_f32x4(globalMax, pPointPos[i]);
       //         globalMin = tfSimdMaxPerElem_f32x4(globalMin, pPointPos[i]);
       //     }


       //     float* calcScale = (float*)tf_malloc(sizeof(float) * mNumOfPoints); 
       //     for (size_t i = 0; i < mNumOfPoints; i++) {
       //         float min3[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
       //         for(size_t ii = 0; ii < mNumOfPoints; ii++) {
       //             if(ii == i)
       //                 continue;
       //             const Tf32x3_s d = {
       //                 gGaussianPoints[i].mPos.x - gGaussianPoints[ii].mPos.x,
       //                 gGaussianPoints[i].mPos.y - gGaussianPoints[ii].mPos.y,
       //                 gGaussianPoints[i].mPos.z - gGaussianPoints[ii].mPos.z,
       //             };
       //             const float dist = d.x * d.x + d.y * d.y + d.z * d.z;
       //             if(dist < min3[0]) {
       //                 min3[0] = min3[1];
       //                 min3[1] = min3[2];
       //                 min3[2] = dist;
       //             }
       //         }
       //         calcScale[i] = (min3[0] + min3[1] + min3[2]) / 3.0f;
       //     }
       //     tf_free(calcScale);
       // }

        //{
        //  for (size_t i = 0; i < mNumOfPoints; i++) {
        //    ((Tf32x3_s*)positionUpdateDesc.pMappedData)[i] = gGaussianPoints[i].mPos;
        //    ((Tf32x3_s*)colorUpdateDesc.pMappedData)[i] = gGaussianPoints[i].mColor;
        //  }
        //}

        BufferLoadDesc ubDesc = {};
        ubDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        ubDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        ubDesc.pData = NULL;
        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            ubDesc.mDesc.pName = "ProjViewUniformBuffer";
            ubDesc.mDesc.mSize = sizeof(UniformBlock);
            ubDesc.ppBuffer = &pProjViewUniformBuffer[i];
            addResource(&ubDesc, NULL);
        }

        // Load fonts
        FontDesc font = {};
        font.pFontPath = "TitilliumText/TitilliumText-Bold.otf";
        fntDefineFonts(&font, 1, &gFontID);

        FontSystemDesc fontRenderDesc = {};
        fontRenderDesc.pRenderer = pRenderer;
        if (!initFontSystem(&fontRenderDesc))
            return false; // report?

        // Initialize Forge User Interface Rendering
        UserInterfaceDesc uiRenderDesc = {};
        uiRenderDesc.pRenderer = pRenderer;
        initUserInterface(&uiRenderDesc);

        // Initialize micro profiler and its UI.
        ProfilerDesc profiler = {};
        profiler.pRenderer = pRenderer;
        profiler.mWidthUI = mSettings.mWidth;
        profiler.mHeightUI = mSettings.mHeight;
        initProfiler(&profiler);

        // Gpu profiler can only be added after initProfile.
        gGpuProfileToken = addGpuProfiler(pRenderer, pGraphicsQueue, "Graphics");

        /************************************************************************/
        // GUI
        /************************************************************************/
        UIComponentDesc guiDesc = {};
        guiDesc.mStartPosition = vec2(mSettings.mWidth * 0.01f, mSettings.mHeight * 0.2f);
        uiCreateComponent(GetName(), &guiDesc, &pGuiWindow);

        if (pRenderer->pProperties->mPipelineStatsQueries)
        {
            static float4     color = { 1.0f, 1.0f, 1.0f, 1.0f };
            DynamicTextWidget statsWidget;
            statsWidget.pText = &gPipelineStats;
            statsWidget.pColor = &color;
            uiCreateComponentWidget(pGuiWindow, "Pipeline Stats", &statsWidget, WIDGET_TYPE_DYNAMIC_TEXT);
        }

        waitForAllResourceLoads();

        CameraMotionParameters cmp{ 60.0f, 20.0f, 200.0f };
        vec3                   camPos{ 10.0f, 10.0f, 20.0f };
        vec3                   lookAt{ vec3(0) };

        pCameraController = initFpsCameraController(camPos, lookAt);

        pCameraController->setMotionParameters(cmp);

        InputSystemDesc inputDesc = {};
        inputDesc.pRenderer = pRenderer;
        inputDesc.pWindow = pWindow;
        inputDesc.pJoystickTexture = "circlepad.tex";
        if (!initInputSystem(&inputDesc))
            return false;

        // App Actions
        InputActionDesc actionDesc = { DefaultInputActions::DUMP_PROFILE_DATA,
                                       [](InputActionContext* ctx)
                                       {
                                           dumpProfileData(((Renderer*)ctx->pUserData)->pName);
                                           return true;
                                       },
                                       pRenderer };
        addInputAction(&actionDesc);
        actionDesc = { DefaultInputActions::TOGGLE_FULLSCREEN,
                       [](InputActionContext* ctx)
                       {
                           WindowDesc* winDesc = ((IApp*)ctx->pUserData)->pWindow;
                           if (winDesc->fullScreen)
                               winDesc->borderlessWindow
                                   ? setBorderless(winDesc, getRectWidth(&winDesc->clientRect), getRectHeight(&winDesc->clientRect))
                                   : setWindowed(winDesc, getRectWidth(&winDesc->clientRect), getRectHeight(&winDesc->clientRect));
                           else
                               setFullscreen(winDesc);
                           return true;
                       },
                       this };
        addInputAction(&actionDesc);
        actionDesc = { DefaultInputActions::EXIT, [](InputActionContext* ctx)
                       {
                           requestShutdown();
                           return true;
                       } };
        addInputAction(&actionDesc);
        InputActionCallback onAnyInput = [](InputActionContext* ctx)
        {
            if (ctx->mActionId > UISystemInputActions::UI_ACTION_START_ID_)
            {
                uiOnInput(ctx->mActionId, ctx->mBool, ctx->pPosition, &ctx->mFloat2);
            }

            return true;
        };

        typedef bool (*CameraInputHandler)(InputActionContext* ctx, DefaultInputActions::DefaultInputAction action);
        static CameraInputHandler onCameraInput = [](InputActionContext* ctx, DefaultInputActions::DefaultInputAction action)
        {
            if (*(ctx->pCaptured))
            {
                float2 delta = uiIsFocused() ? float2(0.f, 0.f) : ctx->mFloat2;
                switch (action)
                {
                case DefaultInputActions::ROTATE_CAMERA:
                    pCameraController->onRotate(delta);
                    break;
                case DefaultInputActions::TRANSLATE_CAMERA:
                    pCameraController->onMove(delta);
                    break;
                case DefaultInputActions::TRANSLATE_CAMERA_VERTICAL:
                    pCameraController->onMoveY(delta[0]);
                    break;
                default:
                    break;
                }
            }
            return true;
        };
        actionDesc = { DefaultInputActions::CAPTURE_INPUT,
                       [](InputActionContext* ctx)
                       {
                           setEnableCaptureInput(!uiIsFocused() && INPUT_ACTION_PHASE_CANCELED != ctx->mPhase);
                           return true;
                       },
                       NULL };
        addInputAction(&actionDesc);
        actionDesc = { DefaultInputActions::ROTATE_CAMERA,
                       [](InputActionContext* ctx) { return onCameraInput(ctx, DefaultInputActions::ROTATE_CAMERA); }, NULL };
        addInputAction(&actionDesc);
        actionDesc = { DefaultInputActions::TRANSLATE_CAMERA,
                       [](InputActionContext* ctx) { return onCameraInput(ctx, DefaultInputActions::TRANSLATE_CAMERA); }, NULL };
        addInputAction(&actionDesc);
        actionDesc = { DefaultInputActions::TRANSLATE_CAMERA_VERTICAL,
                       [](InputActionContext* ctx) { return onCameraInput(ctx, DefaultInputActions::TRANSLATE_CAMERA_VERTICAL); }, NULL };
        addInputAction(&actionDesc);
        actionDesc = { DefaultInputActions::RESET_CAMERA, [](InputActionContext* ctx)
                       {
                           if (!uiWantTextInput())
                               pCameraController->resetView();
                           return true;
                       } };
        addInputAction(&actionDesc);
        GlobalInputActionDesc globalInputActionDesc = { GlobalInputActionDesc::ANY_BUTTON_ACTION, onAnyInput, this };
        setGlobalInputAction(&globalInputActionDesc);

        gFrameIndex = 0;

        return true;
    }

    void Exit()
    {
        exitInputSystem();

        exitCameraController(pCameraController);

        exitUserInterface();

        exitFontSystem();

        // Exit profile
        exitProfiler();

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            removeResource(pProjViewUniformBuffer[i]);
            //removeResource(pSkyboxUniformBuffer[i]);
            if (pRenderer->pProperties->mPipelineStatsQueries)
            {
                removeQueryPool(pRenderer, pPipelineStatsQueryPool[i]);
            }
        }

        removeGpuCmdRing(pRenderer, &gGraphicsCmdRing);
        removeSemaphore(pRenderer, pImageAcquiredSemaphore);

        exitResourceLoaderInterface(pRenderer);

        removeQueue(pRenderer, pGraphicsQueue);

        exitRenderer(pRenderer);
        exitRendererContext(pContext);
        pRenderer = NULL;
        pContext = NULL;
    }

    bool Load(ReloadDesc* pReloadDesc)
    {
        if (pReloadDesc->mType & RELOAD_TYPE_SHADER)
        {
            addShaders();
            addRootSignatures();
            addDescriptorSets();
        }

        if (pReloadDesc->mType & (RELOAD_TYPE_RESIZE | RELOAD_TYPE_RENDERTARGET))
        {
            if (!addSwapChain())
                return false;

            if (!addDepthBuffer())
                return false;
        }

        if (pReloadDesc->mType & (RELOAD_TYPE_SHADER | RELOAD_TYPE_RENDERTARGET))
        {
            
            addPipelines();
        }

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            DescriptorData params[1] = {};
            params[0].pName = "uniformBlock";
            params[0].ppBuffers = &pProjViewUniformBuffer[i];
            updateDescriptorSet(pRenderer, i, pDescriptorSetUniforms, 1, params);
        }

        UserInterfaceLoadDesc uiLoad = {};
        uiLoad.mColorFormat = pSwapChain->ppRenderTargets[0]->mFormat;
        uiLoad.mHeight = mSettings.mHeight;
        uiLoad.mWidth = mSettings.mWidth;
        uiLoad.mLoadType = pReloadDesc->mType;
        loadUserInterface(&uiLoad);

        FontSystemLoadDesc fontLoad = {};
        fontLoad.mColorFormat = pSwapChain->ppRenderTargets[0]->mFormat;
        fontLoad.mHeight = mSettings.mHeight;
        fontLoad.mWidth = mSettings.mWidth;
        fontLoad.mLoadType = pReloadDesc->mType;
        loadFontSystem(&fontLoad);

        initScreenshotInterface(pRenderer, pGraphicsQueue);

        return true;
    }

    void Unload(ReloadDesc* pReloadDesc)
    {
        waitQueueIdle(pGraphicsQueue);

        unloadFontSystem(pReloadDesc->mType);
        unloadUserInterface(pReloadDesc->mType);

        if (pReloadDesc->mType & (RELOAD_TYPE_SHADER | RELOAD_TYPE_RENDERTARGET))
        {
            removePipelines();
            //removeResource(pSphereVertexBuffer);
            //removeResource(pSphereIndexBuffer);
        }

        if (pReloadDesc->mType & (RELOAD_TYPE_RESIZE | RELOAD_TYPE_RENDERTARGET))
        {
            removeSwapChain(pRenderer, pSwapChain);
            removeRenderTarget(pRenderer, pDepthBuffer);
        }

        if (pReloadDesc->mType & RELOAD_TYPE_SHADER)
        {
            removeDescriptorSets();
            removeRootSignatures();
            removeShaders();
        }

        exitScreenshotInterface();
    }

    void Update(float deltaTime)
    {
        updateInputSystem(deltaTime, mSettings.mWidth, mSettings.mHeight);

        pCameraController->update(deltaTime);
        /************************************************************************/
        // Scene Update
        /************************************************************************/
        static float currentTime = 0.0f;
        currentTime += deltaTime * 1000.0f;

        // update camera with time
        mat4 viewMat = pCameraController->getViewMatrix();

        const float  aspectInverse = (float)mSettings.mHeight / (float)mSettings.mWidth;
        const float  horizontal_fov = PI / 2.0f;
        CameraMatrix projMat = CameraMatrix::perspectiveReverseZ(horizontal_fov, aspectInverse, 0.1f, 1000.0f);
        gUniformData.mProjectView = projMat * viewMat;

        viewMat.setTranslation(vec3(0));
        //gUniformDataSky = {};
        //gUniformDataSky.mProjectView = projMat * viewMat;
    }

    void Draw()
    {
        if (pSwapChain->mEnableVsync != mSettings.mVSyncEnabled)
        {
            waitQueueIdle(pGraphicsQueue);
            ::toggleVSync(pRenderer, &pSwapChain);
        }

        uint32_t swapchainImageIndex;
        acquireNextImage(pRenderer, pSwapChain, pImageAcquiredSemaphore, NULL, &swapchainImageIndex);

        RenderTarget*     pRenderTarget = pSwapChain->ppRenderTargets[swapchainImageIndex];
        GpuCmdRingElement elem = getNextGpuCmdRingElement(&gGraphicsCmdRing, true, 1);

        // Stall if CPU is running "gDataBufferCount" frames ahead of GPU
        FenceStatus fenceStatus;
        getFenceStatus(pRenderer, elem.pFence, &fenceStatus);
        if (fenceStatus == FENCE_STATUS_INCOMPLETE)
            waitForFences(pRenderer, 1, &elem.pFence);

        // Update uniform buffers
        BufferUpdateDesc viewProjCbv = { pProjViewUniformBuffer[gFrameIndex] };
        beginUpdateResource(&viewProjCbv);
        memcpy(viewProjCbv.pMappedData, &gUniformData, sizeof(gUniformData));
        endUpdateResource(&viewProjCbv);

        // Reset cmd pool for this frame
        resetCmdPool(pRenderer, elem.pCmdPool);

        if (pRenderer->pProperties->mPipelineStatsQueries)
        {
            QueryData data3D = {};
            QueryData data2D = {};
            getQueryData(pRenderer, pPipelineStatsQueryPool[gFrameIndex], 0, &data3D);
            getQueryData(pRenderer, pPipelineStatsQueryPool[gFrameIndex], 1, &data2D);
            bformat(&gPipelineStats,
                    "\n"
                    "Pipeline Stats 3D:\n"
                    "    VS invocations:      %u\n"
                    "    PS invocations:      %u\n"
                    "    Clipper invocations: %u\n"
                    "    IA primitives:       %u\n"
                    "    Clipper primitives:  %u\n"
                    "\n"
                    "Pipeline Stats 2D UI:\n"
                    "    VS invocations:      %u\n"
                    "    PS invocations:      %u\n"
                    "    Clipper invocations: %u\n"
                    "    IA primitives:       %u\n"
                    "    Clipper primitives:  %u\n",
                    data3D.mPipelineStats.mVSInvocations, data3D.mPipelineStats.mPSInvocations, data3D.mPipelineStats.mCInvocations,
                    data3D.mPipelineStats.mIAPrimitives, data3D.mPipelineStats.mCPrimitives, data2D.mPipelineStats.mVSInvocations,
                    data2D.mPipelineStats.mPSInvocations, data2D.mPipelineStats.mCInvocations, data2D.mPipelineStats.mIAPrimitives,
                    data2D.mPipelineStats.mCPrimitives);
        }

        Cmd* cmd = elem.pCmds[0];
        beginCmd(cmd);

        cmdBeginGpuFrameProfile(cmd, gGpuProfileToken);
        if (pRenderer->pProperties->mPipelineStatsQueries)
        {
            cmdResetQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], 0, 2);
            QueryDesc queryDesc = { 0 };
            cmdBeginQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
        }

        RenderTargetBarrier barriers[] = {
            { pRenderTarget, RESOURCE_STATE_PRESENT, RESOURCE_STATE_RENDER_TARGET },
        };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Skybox/Planets");

        // simply record the screen cleaning command
        BindRenderTargetsDesc bindRenderTargets = {};
        bindRenderTargets.mRenderTargetCount = 1;
        bindRenderTargets.mRenderTargets[0] = { pRenderTarget, LOAD_ACTION_CLEAR };
        bindRenderTargets.mDepthStencil = { pDepthBuffer, LOAD_ACTION_CLEAR };
        cmdBindRenderTargets(cmd, &bindRenderTargets);
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 0.0f, 1.0f);
        cmdSetScissor(cmd, 0, 0, pRenderTarget->mWidth, pRenderTarget->mHeight);

        //const uint32_t skyboxVbStride = sizeof(float) * 4;
        //// draw skybox
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Gaussian Points");
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 1.0f, 1.0f);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetUniforms);
        {
            Buffer*  bufferArgs[2] = { pGaussianPosition, pGaussianColor };
            uint32_t strideArgs[2] = { sizeof(struct Tf32x3_s), sizeof(struct Tf32x3_s) };
            cmdBindVertexBuffer(cmd, 2, bufferArgs, strideArgs, NULL);
        }
        cmdBindPipeline(cmd, pParticlePipeline);
        cmdDraw(cmd, mNumOfPoints, 0);
        
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 0.0f, 1.0f);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken); // Draw Skybox/Planets
        cmdBindRenderTargets(cmd, NULL);

        if (pRenderer->pProperties->mPipelineStatsQueries)
        {
            QueryDesc queryDesc = { 0 };
            cmdEndQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);

            queryDesc = { 1 };
            cmdBeginQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
        }

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw UI");

        bindRenderTargets = {};
        bindRenderTargets.mRenderTargetCount = 1;
        bindRenderTargets.mRenderTargets[0] = { pRenderTarget, LOAD_ACTION_LOAD };
        bindRenderTargets.mDepthStencil = { NULL, LOAD_ACTION_DONTCARE };
        cmdBindRenderTargets(cmd, &bindRenderTargets);

        gFrameTimeDraw.mFontColor = 0xff00ffff;
        gFrameTimeDraw.mFontSize = 18.0f;
        gFrameTimeDraw.mFontID = gFontID;
        float2 txtSizePx = cmdDrawCpuProfile(cmd, float2(8.f, 15.f), &gFrameTimeDraw);
        cmdDrawGpuProfile(cmd, float2(8.f, txtSizePx.y + 75.f), gGpuProfileToken, &gFrameTimeDraw);

        cmdDrawUserInterface(cmd);

        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        cmdBindRenderTargets(cmd, NULL);

        barriers[0] = { pRenderTarget, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_PRESENT };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);

        cmdEndGpuFrameProfile(cmd, gGpuProfileToken);

        if (pRenderer->pProperties->mPipelineStatsQueries)
        {
            QueryDesc queryDesc = { 1 };
            cmdEndQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
            cmdResolveQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], 0, 2);
        }

        endCmd(cmd);

        FlushResourceUpdateDesc flushUpdateDesc = {};
        flushUpdateDesc.mNodeIndex = 0;
        flushResourceUpdates(&flushUpdateDesc);
        Semaphore* waitSemaphores[2] = { flushUpdateDesc.pOutSubmittedSemaphore, pImageAcquiredSemaphore };

        QueueSubmitDesc submitDesc = {};
        submitDesc.mCmdCount = 1;
        submitDesc.mSignalSemaphoreCount = 1;
        submitDesc.mWaitSemaphoreCount = TF_ARRAY_COUNT(waitSemaphores);
        submitDesc.ppCmds = &cmd;
        submitDesc.ppSignalSemaphores = &elem.pSemaphore;
        submitDesc.ppWaitSemaphores = waitSemaphores;
        submitDesc.pSignalFence = elem.pFence;
        queueSubmit(pGraphicsQueue, &submitDesc);

        QueuePresentDesc presentDesc = {};
        presentDesc.mIndex = swapchainImageIndex;
        presentDesc.mWaitSemaphoreCount = 1;
        presentDesc.pSwapChain = pSwapChain;
        presentDesc.ppWaitSemaphores = &elem.pSemaphore;
        presentDesc.mSubmitDone = true;

        queuePresent(pGraphicsQueue, &presentDesc);
        flipProfiler();

        gFrameIndex = (gFrameIndex + 1) % gDataBufferCount;
    }

    const char* GetName() { return "01_Transformations"; }

    bool addSwapChain()
    {
        SwapChainDesc swapChainDesc = {};
        swapChainDesc.mWindowHandle = pWindow->handle;
        swapChainDesc.mPresentQueueCount = 1;
        swapChainDesc.ppPresentQueues = &pGraphicsQueue;
        swapChainDesc.mWidth = mSettings.mWidth;
        swapChainDesc.mHeight = mSettings.mHeight;
        swapChainDesc.mImageCount = getRecommendedSwapchainImageCount(pRenderer, &pWindow->handle);
        swapChainDesc.mColorFormat = getSupportedSwapchainFormat(pRenderer, &swapChainDesc, COLOR_SPACE_SDR_SRGB);
        swapChainDesc.mColorSpace = COLOR_SPACE_SDR_SRGB;
        swapChainDesc.mEnableVsync = mSettings.mVSyncEnabled;
        swapChainDesc.mFlags = SWAP_CHAIN_CREATION_FLAG_ENABLE_FOVEATED_RENDERING_VR;
        ::addSwapChain(pRenderer, &swapChainDesc, &pSwapChain);

        return pSwapChain != NULL;
    }

    bool addDepthBuffer()
    {
        // Add depth buffer
        RenderTargetDesc depthRT = {};
        depthRT.mArraySize = 1;
        depthRT.mClearValue.depth = 0.0f;
        depthRT.mClearValue.stencil = 0;
        depthRT.mDepth = 1;
        depthRT.mFormat = TinyImageFormat_D32_SFLOAT;
        depthRT.mStartState = RESOURCE_STATE_DEPTH_WRITE;
        depthRT.mHeight = mSettings.mHeight;
        depthRT.mSampleCount = SAMPLE_COUNT_1;
        depthRT.mSampleQuality = 0;
        depthRT.mWidth = mSettings.mWidth;
        depthRT.mFlags = TEXTURE_CREATION_FLAG_ON_TILE | TEXTURE_CREATION_FLAG_VR_MULTIVIEW;
        addRenderTarget(pRenderer, &depthRT, &pDepthBuffer);

        return pDepthBuffer != NULL;
    }

    void addDescriptorSets()
    {
        DescriptorSetDesc desc = { pRootSignature, DESCRIPTOR_UPDATE_FREQ_PER_FRAME, gDataBufferCount * 2 };
        addDescriptorSet(pRenderer, &desc, &pDescriptorSetUniforms);
    }

    void removeDescriptorSets()
    {
        removeDescriptorSet(pRenderer, pDescriptorSetUniforms);
    }

    void addRootSignatures()
    {
        Shader*  shaders[2];
        uint32_t shadersCount = 0;
        shaders[shadersCount++] = pParticleShader;

        RootSignatureDesc rootDesc = {};
        rootDesc.mShaderCount = shadersCount;
        rootDesc.ppShaders = shaders;
        addRootSignature(pRenderer, &rootDesc, &pRootSignature);
    }

    void removeRootSignatures() { removeRootSignature(pRenderer, pRootSignature); }

    void addShaders()
    {

        ShaderLoadDesc particleShader = {};
        particleShader.mStages[0].pFileName = "particle.vert";
        particleShader.mStages[1].pFileName = "particle.frag";
        addShader(pRenderer, &particleShader, &pParticleShader);
    }

    void removeShaders() { removeShader(pRenderer, pParticleShader); }

    void addPipelines() {

        {
            RasterizerStateDesc rasterizerStateDesc = {};
            rasterizerStateDesc.mCullMode = CULL_MODE_NONE;

            DepthStateDesc depthStateDesc = {};
            depthStateDesc.mDepthTest = true;
            depthStateDesc.mDepthWrite = false;
            depthStateDesc.mDepthFunc = CMP_GEQUAL;

            VertexLayout vertexLayout = {};
            vertexLayout.mBindingCount = 2;
            vertexLayout.mAttribCount = 2;
            vertexLayout.mBindings[0].mStride = sizeof(struct Tf32x3_s);
            vertexLayout.mBindings[1].mStride = sizeof(struct Tf32x3_s);

            vertexLayout.mAttribs[0].mSemantic = SEMANTIC_POSITION;
            vertexLayout.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            vertexLayout.mAttribs[0].mBinding = 0;
            vertexLayout.mAttribs[0].mLocation = 0;
            vertexLayout.mAttribs[0].mOffset = 0;

            vertexLayout.mAttribs[1].mSemantic = SEMANTIC_TEXCOORD0;
            vertexLayout.mAttribs[1].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            vertexLayout.mAttribs[1].mBinding = 1;
            vertexLayout.mAttribs[1].mLocation = 1;
            vertexLayout.mAttribs[1].mOffset = 0;

            PipelineDesc desc = {};
            desc.mType = PIPELINE_TYPE_GRAPHICS;
            GraphicsPipelineDesc& pipelineSettings = desc.mGraphicsDesc;
            pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_POINT_LIST;
            pipelineSettings.mRenderTargetCount = 1;
            pipelineSettings.pDepthState = &depthStateDesc;
            pipelineSettings.pColorFormats = &pSwapChain->ppRenderTargets[0]->mFormat;
            pipelineSettings.mSampleCount = pSwapChain->ppRenderTargets[0]->mSampleCount;
            pipelineSettings.mSampleQuality = pSwapChain->ppRenderTargets[0]->mSampleQuality;
            pipelineSettings.mDepthStencilFormat = pDepthBuffer->mFormat;
            pipelineSettings.pRootSignature = pRootSignature;
            pipelineSettings.pShaderProgram = pParticleShader;
            pipelineSettings.pVertexLayout = &vertexLayout;
            pipelineSettings.pRasterizerState = &rasterizerStateDesc;
            pipelineSettings.mVRFoveatedRendering = true;
            addPipeline(pRenderer, &desc, &pParticlePipeline);
        }

    }

    void removePipelines()
    {
        //removePipeline(pRenderer, pSkyBoxDrawPipeline);
        removePipeline(pRenderer, pParticlePipeline);
    }
};
DEFINE_APPLICATION_MAIN(Transformations)
