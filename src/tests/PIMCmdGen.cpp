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

#include "tests/PIMCmdGen.h"

#include <memory>

vector<PIMCmd> PIMCmdGen::getPIMCmds(KernelType ktype, int num_jump_to_be_taken,
                                     int num_jump_to_be_taken_odd_bank,
                                     int num_jump_to_be_taken_even_bank)
{
    unique_ptr<IPIMCmd> pim_kernel = nullptr;
    switch (ktype)
    {
        /*
        case KernelType::BN:
            pim_kernel = make_unique<BatchNormPIMKernel>(ktype);
            break;
        */
        case KernelType::RELU:
            pim_kernel = make_unique<ActPIMKernel>(ktype);
            break;
        case KernelType::MUL:
            pim_kernel = make_unique<EltwisePIMKernel>(ktype);
            break;
        case KernelType::ADD:
            pim_kernel = make_unique<EltwisePIMKernel>(ktype);
            break;
        case KernelType::GEMV:
            pim_kernel = make_unique<GemvPIMKernel>(ktype);
            break;
        case KernelType::GEMVTREE:
            pim_kernel = make_unique<GemvPIMKernel>(ktype);
            break;
        //我改的代码
        case KernelType::BN:
            pim_kernel = make_unique<BnPIMKernel>(ktype);
            break;
        default:
            throw invalid_argument("Invalid kernel type");
        
    }
    return pim_kernel->generateKernel(num_jump_to_be_taken, num_jump_to_be_taken_odd_bank,
                                      num_jump_to_be_taken_even_bank);
}


/***************************************************************************************************
 * 4) GemvPIMKernel: "指令流水优化"重点
 *    - 合并MAC指令 => repeat=8
 *    - 减少NOP => 由7改为2
 *    - 改进JUMP => 仅在非0时push
 **************************************************************************************************/
vector<PIMCmd> GemvPIMKernel::generateKernel(int num_jump_to_be_taken,
                                             int num_jump_to_be_taken_odd_bank,
                                             int num_jump_to_be_taken_even_bank)
{
    vector<PIMCmd> pim_cmds;
    if(kernelType==KernelType::GEMV)
    {
        // 合并MAC => repeat=8, 减少NOP => 2, 改进JUMP => 只在非0时push
        vector<PIMCmd> tmp_cmds{
            // even MAC *8
            PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                   PIMOpdType::EVEN_BANK, 8),
            // jumpEven
            PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken_even_bank, 2),

            // odd MAC *8
            PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                   PIMOpdType::ODD_BANK, 8),
            // jumpOdd
            PIMCmd(PIMCmdType::JUMP, num_jump_to_be_taken_odd_bank, 2),

            // 减少NOP => 2
            PIMCmd(PIMCmdType::NOP, 2)
        };
        pim_cmds.insert(pim_cmds.end(), tmp_cmds.begin(), tmp_cmds.end());
    }
    else if(kernelType==KernelType::GEMVTREE)
    {
        // 你原先的 gemvtree 逻辑
        vector<PIMCmd> tmp_cmds{
            PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B,PIMOpdType::GRF_A,PIMOpdType::EVEN_BANK,1),
            PIMCmd(PIMCmdType::JUMP,7,2),
            PIMCmd(PIMCmdType::NOP,7),
            PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_B,PIMOpdType::GRF_B,PIMOpdType::EVEN_BANK,1),
            PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B,PIMOpdType::GRF_A,PIMOpdType::ODD_BANK,1),
            PIMCmd(PIMCmdType::JUMP,7,2),
            PIMCmd(PIMCmdType::NOP,7),
            PIMCmd(PIMCmdType::MUL, PIMOpdType::GRF_B,PIMOpdType::GRF_B,PIMOpdType::EVEN_BANK,1)
        };
        pim_cmds.insert(pim_cmds.end(), tmp_cmds.begin(), tmp_cmds.end());
    }
    else
    {
        throw invalid_argument("Not supported gemv operation");
    }

    if(num_jump_to_be_taken!=0)
    {
        pim_cmds.push_back(
            PIMCmd(PIMCmdType::JUMP,num_jump_to_be_taken,(int)pim_cmds.size()+1)
        );
    }
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT,0));
    return pim_cmds;
}
