/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#ifndef __TEST_CASES_H__
#define __TEST_CASES_H__

#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

#include "Burst.h"
#include "FP16.h"
#include "gtest/gtest.h"
#include "tests/PIMKernel.h"

using namespace DRAMSim;

class basicFixture : public testing::Test
{
  public:
    basicFixture() {}
    ~basicFixture() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class MemBandwidthFixture : public testing::Test
{
  public:
    MemBandwidthFixture() {}
    ~MemBandwidthFixture() {}
    virtual void SetUp()
    {
        cur_cycle = 0;
        mem = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                    "system_hbm.ini", ".", "example_app", 256 * 16);
        mem_size = (uint64_t)getConfigParam(UINT, "NUM_CHANS") * getConfigParam(UINT, "NUM_RANKS") *
                   getConfigParam(UINT, "NUM_BANK_GROUPS") * getConfigParam(UINT, "NUM_BANKS") *
                   getConfigParam(UINT, "NUM_ROWS") * getConfigParam(UINT, "NUM_COLS");
        basic_stride =
            (getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8);
    }
    virtual void TearDown()
    {
        printResult(cur_cycle);
    }

    uint64_t measureCycle(bool is_write)
    {
        printTestMessage();
        write_ = is_write;
        generateMemTraffic(is_write);

        while (mem->hasPendingTransactions())
        {
            cur_cycle++;
            mem->update();
        }

        return cur_cycle;
    }

    void printTestMessage()
    {
        cout << ">> Bandwidth Test" << endl;
        cout << "  Num. channels: " << getConfigParam(UINT, "NUM_CHANS") << endl;
        cout << "  Num. ranks: " << getConfigParam(UINT, "NUM_RANKS") << endl;
        cout << "  Num. banks: " << getConfigParam(UINT, "NUM_BANKS") << endl;
        cout << "  Data size (byte): " << data_size_in_byte << endl;
    }

    void printResult(uint64_t cycle)
    {
        uint64_t totalReads = 0;
        uint64_t totalWrites = 0;

        for (size_t i = 0; i < getConfigParam(UINT, "NUM_CHANS"); i++)
        {
            MemoryController* mem_ctrl = mem->channels[i]->memoryController;
            totalReads += mem_ctrl->totalReads;
            totalWrites += mem_ctrl->totalWrites;
        }
        uint32_t bw = (totalReads + totalWrites) * getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") *
                      getConfigParam(UINT, "BL") / 8 / (cycle * getConfigParam(FLOAT, "tCK"));
        cout << endl;
        cout << "> Test Result " << endl;
        cout << "> BW (GB/s): " << bw << endl;
    }

    void setDataSize(unsigned size)
    {
        data_size_in_byte = size;
    }

    void generateMemTraffic(bool is_write)
    {
        int num_trans = 0;
        BurstType nullBst;

        for (uint64_t i = 0; i < mem_size; ++i)
        {
            if (num_trans >= (data_size_in_byte / basic_stride))
            {
                break;
            }
            uint64_t addr = i * basic_stride;
            mem->addTransaction(is_write, addr, &nullBst);
            num_trans++;
        }
    }

  private:
    bool write_;
    uint64_t cur_cycle = 0;
    uint64_t mem_size;
    uint64_t data_size_in_byte;
    uint64_t basic_stride;
    shared_ptr<MultiChannelMemorySystem> mem;
};

class DataDim
{
  private:
    unsigned getPrecisionToByte()
    {
        switch (PIMConfiguration::getPIMPrecision())
        {
            case INT8:
                return 1;
            case FP16:
                return 2;
            case FP32:
                return 4;
            default:
                return 0;
        }
    }

    void loadData(KernelType kn_type)
    {
            string input_dim_str = to_string(input_dim_);

            switch (kn_type)
            {
                case KernelType::GEMV:
                case KernelType::GEMVTREE:
                {
                    string output_dim_str = to_string(output_dim_);

                    string in_out_dim_str = output_dim_str + "x" + input_dim_str;

                    if (batch_size_ > 1)
                    {
                        string batch_size_str = to_string(batch_size_);
                        string batch_in_out_dim_str = batch_size_str + "_" + in_out_dim_str;
                        input_npbst_.loadFp16("data/gemv/gemv_input_batch_" + batch_in_out_dim_str +
                                            ".npy");
                        weight_npbst_.loadFp16("data/gemv/gemv_weight_batch_" + batch_in_out_dim_str +
                                            ".npy");
                        output_npbst_.loadFp16("data/gemv/gemv_output_batch_" + batch_in_out_dim_str +
                                            ".npy");
                    }
                    else
                    {
                        input_npbst_.loadFp16("data/gemv/gemv_input_" + in_out_dim_str + ".npy");
                        weight_npbst_.loadFp16("data/gemv/gemv_weight_" + in_out_dim_str + ".npy");
                        output_npbst_.loadFp16("data/gemv/gemv_output_" + in_out_dim_str + ".npy");
                    }

                    // output_dim_ = weight_npbst_.bShape[0];
                    output_dim_ = bShape1ToDim(output_npbst_.bShape[1]);
                    input_dim_ = bShape1ToDim(input_npbst_.bShape[1]);
                    batch_size_ = input_npbst_.bShape[0];
                    return;
                }
                case KernelType::ADD:
                {
                    input_npbst_.loadFp16("data/add/resadd_input0_" + input_dim_str + ".npy");
                    input1_npbst_.loadFp16("data/add/resadd_input1_" + input_dim_str + ".npy");
                    output_npbst_.loadFp16("data/add/resadd_output_" + input_dim_str + ".npy");

                    output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
                    input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
                    input1_dim_ = bShape1ToDim(input1_npbst_.getTotalDim());

                    return;
                }
                case KernelType::MUL:
                {
                    input_npbst_.loadFp16("data/mul/eltmul_input0_" + input_dim_str + ".npy");
                    input1_npbst_.loadFp16("data/mul/eltmul_input1_" + input_dim_str + ".npy");
                    output_npbst_.loadFp16("data/mul/eltmul_output_" + input_dim_str + ".npy");

                    output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
                    input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
                    input1_dim_ = bShape1ToDim(input1_npbst_.getTotalDim());

                    return;
                }
                case KernelType::RELU:
                {
                    input_npbst_.loadFp16("data/relu/relu_input_" + input_dim_str + ".npy");
                    output_npbst_.loadFp16("data/relu/relu_output_" + input_dim_str + ".npy");

                    output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
                    input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());

                    return;
                }
                //我改的代码

                case KernelType::BN:
                {
                    // BN 操作需要加载输入 x、scale、shift 和输出 y
                    input_npbst_.loadFp16("data/bn/bn_input_x_" + input_dim_str + ".npy");
                    input1_npbst_.loadFp16("data/bn/bn_scale_" + input_dim_str + ".npy");
                    weight_npbst_.loadFp16("data/bn/bn_shift_" + input_dim_str + ".npy");
                    output_npbst_.loadFp16("data/bn/bn_output_y_" + input_dim_str + ".npy");

                    // 假设 BN 的输出维度与输入维度相同
                    output_dim_ = bShape1ToDim(output_npbst_.getTotalDim());
                    input_dim_ = bShape1ToDim(input_npbst_.getTotalDim());
                    batch_size_ = input_npbst_.bShape[0];
                    return;
                }
                default:
                {
                    ERROR("== Error - Unknown KernelType trying to load data");
                    exit(-1);
                    return;
                }
            }
        //}
    }

    void loadDummyData(KernelType kn_type)
    {
            switch (kn_type)
            {
                case KernelType::GEMV:
                case KernelType::GEMVTREE:
                {
                    weight_npbst_.shape.push_back(output_dim_);
                    weight_npbst_.shape.push_back(input_dim_);
                    weight_npbst_.loadTobShape(16);

                    input_npbst_.shape.push_back(batch_size_);
                    input_npbst_.shape.push_back(input_dim_);
                    input_npbst_.loadTobShape(16);

                    for (int i = 0; i < input_npbst_.bShape[1]; i++)
                    {
                        BurstType null_bst;
                        null_bst.set((float)0);
                        input_npbst_.bData.push_back(null_bst);
                    }

                    return;
                }
                case KernelType::ADD:
                case KernelType::MUL:
                case KernelType::RELU:
                {
                    input_npbst_.shape.push_back(batch_size_);
                    input_npbst_.shape.push_back(input_dim_);
                    input_npbst_.loadTobShape(16);

                    output_npbst_.shape.push_back(batch_size_);
                    output_npbst_.shape.push_back(output_dim_);
                    output_npbst_.loadTobShape(16);

                    return;
                }
                //我改的代码
                case KernelType::BN:
                {
                    // 1. Initialize input x: batch_size_ x input_dim_
                    input_npbst_.shape.push_back(batch_size_);
                    input_npbst_.shape.push_back(input_dim_);
                    input_npbst_.loadTobShape(16); // Assuming burst size of 16
                    for (int i = 0; i < input_npbst_.getTotalDim(); i++)
                    {
                        BurstType bst;
                        bst.set(1.0f);  // Set x to 1.0
                        input_npbst_.bData.push_back(bst);
                    }

                    // 2. Initialize input1_npbst_ as gamma: 1 x input_dim_
                    input1_npbst_.shape.push_back(1);
                    input1_npbst_.shape.push_back(input_dim_);
                    input1_npbst_.loadTobShape(16);
                    for (int i = 0; i < input1_npbst_.getTotalDim(); i++)
                    {
                        BurstType bst;
                        bst.set(1.0f);  // Set gamma to 1.0
                        input1_npbst_.bData.push_back(bst);
                    }

                    // 3. Initialize weight_npbst_ as shift (beta): 1 x input_dim_
                    weight_npbst_.shape.push_back(1);
                    weight_npbst_.shape.push_back(input_dim_);
                    weight_npbst_.loadTobShape(16);
                    for (int i = 0; i < weight_npbst_.getTotalDim(); i++)
                    {
                        BurstType bst;
                        bst.set(0.0f);  // Set shift (beta) to 0.0
                        weight_npbst_.bData.push_back(bst);
                    }

                    // 4. Initialize output y: batch_size_ x input_dim_
                    output_npbst_.shape.push_back(batch_size_);
                    output_npbst_.shape.push_back(input_dim_);
                    output_npbst_.loadTobShape(16);
                    for (int i = 0; i < output_npbst_.getTotalDim(); i++)
                    {
                        BurstType bst;
                        bst.set(1.0f);  // Set y to 1.0
                        output_npbst_.bData.push_back(bst);
                    }

                    // 5. Initialize mean: 1 x input_dim_
                    mean_npbst_.shape.push_back(1);
                    mean_npbst_.shape.push_back(input_dim_);
                    mean_npbst_.loadTobShape(16);
                    for (int i = 0; i < mean_npbst_.getTotalDim(); i++)
                    {
                        BurstType bst;
                        bst.set(0.0f);  // Set mean to 0.0
                        mean_npbst_.bData.push_back(bst);
                    }

                    // 6. Initialize var: 1 x input_dim_
                    var_npbst_.shape.push_back(1);
                    var_npbst_.shape.push_back(input_dim_);
                    var_npbst_.loadTobShape(16);
                    for (int i = 0; i < var_npbst_.getTotalDim(); i++)
                    {
                        BurstType bst;
                        bst.set(1.0f);  // Set variance to 1.0
                        var_npbst_.bData.push_back(bst);
                    }

                    // 7. Initialize gamma_npbst_ as gamma: 1 x input_dim_
                    gamma_npbst_.shape.push_back(1);
                    gamma_npbst_.shape.push_back(input_dim_);
                    gamma_npbst_.loadTobShape(16);
                    for (int i = 0; i < gamma_npbst_.getTotalDim(); i++)
                    {
                        BurstType bst;
                        bst.set(1.0f);  // Set gamma_npbst_ to 1.0
                        gamma_npbst_.bData.push_back(bst);
                    }

                    // 8. Initialize beta_npbst_ as beta: 1 x input_dim_
                    beta_npbst_.shape.push_back(1);
                    beta_npbst_.shape.push_back(input_dim_);
                    beta_npbst_.loadTobShape(16);
                    for (int i = 0; i < beta_npbst_.getTotalDim(); i++)
                    {
                        BurstType bst;
                        bst.set(0.0f);  // Set beta_npbst_ to 0.0
                        beta_npbst_.bData.push_back(bst);
                    }

                    return;
                }
                // 添加对 GEMM 的支持
                default:
                {
                    return;
                }
            }
    }

  public:
    /* data */
    NumpyBurstType input_npbst_;
    NumpyBurstType input1_npbst_;
    NumpyBurstType weight_npbst_;
    NumpyBurstType output_npbst_;

    /* dump */
    NumpyBurstType preloaded_npbst_;
    NumpyBurstType result_npbst_;
    NumpyBurstType reduced_result_npbst_;

    size_t burst_cnt_;
    BurstType* preloaded_bst_;
    BurstType* result_;
    BurstType* reduced_result_;

    /* dim */
    unsigned long output_dim_;
    int input_dim_;
    int input1_dim_;
    int batch_size_;
    bool used_data_;

    //我改的代码
    //unsigned M_;
    //unsigned N_;
    //unsigned K_;
    // BN相关变量
    NumpyBurstType mean_npbst_;
    NumpyBurstType var_npbst_;
    NumpyBurstType gamma_npbst_;
    NumpyBurstType beta_npbst_;

    DataDim(KernelType kn_type, uint32_t batch_size, uint32_t output_dim, uint32_t input_dim,
            bool used_data)
    {
        batch_size_ = batch_size;
        output_dim_ = output_dim;
        input_dim_ = input_dim;
        used_data_ = used_data;

        switch (kn_type)
        {
            case KernelType::MUL:
            case KernelType::ADD:
            {
                input1_dim_ = input_dim;
                break;
            }
            case KernelType::BN:
            {
                // BN 不需要额外的 input1_dim_
                input1_dim_ = output_dim; // scale 和 shift 的维度与输出维度相同
                break;
            }
            default:
            {
                break;
            }
        }

        // load data from files
        if (used_data_)
            loadData(kn_type);
        else
            loadDummyData(kn_type);
    }




    uint32_t getDataSize(uint32_t dim1, uint32_t dim2 = 1, uint32_t dim3 = 1)
    {
        return dim1 * dim2 * dim3 * getPrecisionToByte();
    }

    void printDim(KernelType kn_type)
    {
       
            switch (kn_type)
            {
                case KernelType::GEMV:
                case KernelType::GEMVTREE:
                {
                    cout << "  Weight data dimension : " << output_dim_ << "x" << input_dim_ << endl;
                    if (batch_size_ > 1)
                    {
                        cout << "  Input data dimension : " << input_dim_ << "x" << batch_size_ << endl;
                        cout << "  Output data dimension : " << output_dim_ << "x" << batch_size_
                            << endl;
                    }
                    else
                    {
                        cout << "  Input data dimension : " << input_dim_ << endl;
                        cout << "  Output data dimension : " << output_dim_ << endl;
                    }
                    break;
                }
                case KernelType::MUL:
                case KernelType::ADD:
                case KernelType::RELU:
                {
                    cout << "  Input/output data dimension : " << output_dim_ << endl;
                    break;
                }
                case KernelType::BN:  // 为BN添加输出维度信息
                {
                    cout << "  Batch Norm Input data dimension : " << input_dim_ << "x" << batch_size_ << endl;
                    cout << "  Output data dimension : " << output_dim_ << "x" << batch_size_ << endl;
                    cout << "  Mean/Variance dimension : " << 1 << "x" << output_dim_ << endl;
                    cout << "  Gamma/Beta dimension : " << 1 << "x" << output_dim_ << endl;
                    break;
                }
                default:
                {
                    ERROR("== Error - Unknown KernelType trying to load data");
                    exit(-1);
                    break;
                }
            }
        //}
    }
    uint32_t getNumElementsPerBlocks()
    {
        return ((getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8) /
                getPrecisionToByte());
    }

    uint32_t dimTobShape(int in_dim)
    {
        return ceil(in_dim / getNumElementsPerBlocks());
    }

    uint32_t bShape1ToDim(int bSahpe1)
    {
        return bSahpe1 * getNumElementsPerBlocks();
    }
};

#endif
