/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "turbine_model/turbine_model.h"





#include "acados_solver_turbine.h"

#define NX     TURBINE_NX
#define NZ     TURBINE_NZ
#define NU     TURBINE_NU
#define NP     TURBINE_NP
#define NP_GLOBAL     TURBINE_NP_GLOBAL
#define NY0    TURBINE_NY0
#define NY     TURBINE_NY
#define NYN    TURBINE_NYN

#define NBX    TURBINE_NBX
#define NBX0   TURBINE_NBX0
#define NBU    TURBINE_NBU
#define NG     TURBINE_NG
#define NBXN   TURBINE_NBXN
#define NGN    TURBINE_NGN

#define NH     TURBINE_NH
#define NHN    TURBINE_NHN
#define NH0    TURBINE_NH0
#define NPHI   TURBINE_NPHI
#define NPHIN  TURBINE_NPHIN
#define NPHI0  TURBINE_NPHI0
#define NR     TURBINE_NR

#define NS     TURBINE_NS
#define NS0    TURBINE_NS0
#define NSN    TURBINE_NSN

#define NSBX   TURBINE_NSBX
#define NSBU   TURBINE_NSBU
#define NSH0   TURBINE_NSH0
#define NSH    TURBINE_NSH
#define NSHN   TURBINE_NSHN
#define NSG    TURBINE_NSG
#define NSPHI0 TURBINE_NSPHI0
#define NSPHI  TURBINE_NSPHI
#define NSPHIN TURBINE_NSPHIN
#define NSGN   TURBINE_NSGN
#define NSBXN  TURBINE_NSBXN



// ** solver data **

turbine_solver_capsule * turbine_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(turbine_solver_capsule));
    turbine_solver_capsule *capsule = (turbine_solver_capsule *) capsule_mem;

    return capsule;
}


int turbine_acados_free_capsule(turbine_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int turbine_acados_create(turbine_solver_capsule* capsule)
{
    int N_shooting_intervals = TURBINE_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return turbine_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int turbine_acados_update_time_steps(turbine_solver_capsule* capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "turbine_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;
}

/**
 * Internal function for turbine_acados_create: step 1
 */
void turbine_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    nlp_solver_plan->nlp_cost[0] = LINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = IRK;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* turbine_acados_create_setup_dimensions(turbine_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    nbxe[0] = 3;
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);

    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for turbine_acados_create: step 3
 */
void turbine_acados_create_setup_functions(turbine_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;




    // implicit dae
    capsule->impl_dae_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_fun[i], turbine_impl_dae_fun);
    }

    capsule->impl_dae_fun_jac_x_xdot_z = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_fun_jac_x_xdot_z[i], turbine_impl_dae_fun_jac_x_xdot_z);
    }

    capsule->impl_dae_jac_x_xdot_u_z = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_jac_x_xdot_u_z[i], turbine_impl_dae_jac_x_xdot_u_z);
    }


#undef MAP_CASADI_FNC
}


/**
 * Internal function for turbine_acados_create: step 4
 */
void turbine_acados_create_set_default_parameters(turbine_solver_capsule* capsule)
{

    // no parameters defined


    // no global parameters defined
}


/**
 * Internal function for turbine_acados_create: step 5
 */
void turbine_acados_setup_nlp_in(turbine_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
//    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
//    capsule->nlp_in = nlp_in;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        turbine_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    double time_step = 0.05;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.05;
        cost_scaling[1] = 0.05;
        cost_scaling[2] = 0.05;
        cost_scaling[3] = 0.05;
        cost_scaling[4] = 0.05;
        cost_scaling[5] = 0.05;
        cost_scaling[6] = 0.05;
        cost_scaling[7] = 0.05;
        cost_scaling[8] = 0.05;
        cost_scaling[9] = 0.05;
        cost_scaling[10] = 0.05;
        cost_scaling[11] = 0.05;
        cost_scaling[12] = 0.05;
        cost_scaling[13] = 0.05;
        cost_scaling[14] = 0.05;
        cost_scaling[15] = 0.05;
        cost_scaling[16] = 0.05;
        cost_scaling[17] = 0.05;
        cost_scaling[18] = 0.05;
        cost_scaling[19] = 0.05;
        cost_scaling[20] = 0.05;
        cost_scaling[21] = 0.05;
        cost_scaling[22] = 0.05;
        cost_scaling[23] = 0.05;
        cost_scaling[24] = 0.05;
        cost_scaling[25] = 0.05;
        cost_scaling[26] = 0.05;
        cost_scaling[27] = 0.05;
        cost_scaling[28] = 0.05;
        cost_scaling[29] = 0.05;
        cost_scaling[30] = 0.05;
        cost_scaling[31] = 0.05;
        cost_scaling[32] = 0.05;
        cost_scaling[33] = 0.05;
        cost_scaling[34] = 0.05;
        cost_scaling[35] = 0.05;
        cost_scaling[36] = 0.05;
        cost_scaling[37] = 0.05;
        cost_scaling[38] = 0.05;
        cost_scaling[39] = 0.05;
        cost_scaling[40] = 0.05;
        cost_scaling[41] = 0.05;
        cost_scaling[42] = 0.05;
        cost_scaling[43] = 0.05;
        cost_scaling[44] = 0.05;
        cost_scaling[45] = 0.05;
        cost_scaling[46] = 0.05;
        cost_scaling[47] = 0.05;
        cost_scaling[48] = 0.05;
        cost_scaling[49] = 0.05;
        cost_scaling[50] = 0.05;
        cost_scaling[51] = 0.05;
        cost_scaling[52] = 0.05;
        cost_scaling[53] = 0.05;
        cost_scaling[54] = 0.05;
        cost_scaling[55] = 0.05;
        cost_scaling[56] = 0.05;
        cost_scaling[57] = 0.05;
        cost_scaling[58] = 0.05;
        cost_scaling[59] = 0.05;
        cost_scaling[60] = 0.05;
        cost_scaling[61] = 0.05;
        cost_scaling[62] = 0.05;
        cost_scaling[63] = 0.05;
        cost_scaling[64] = 0.05;
        cost_scaling[65] = 0.05;
        cost_scaling[66] = 0.05;
        cost_scaling[67] = 0.05;
        cost_scaling[68] = 0.05;
        cost_scaling[69] = 0.05;
        cost_scaling[70] = 0.05;
        cost_scaling[71] = 0.05;
        cost_scaling[72] = 0.05;
        cost_scaling[73] = 0.05;
        cost_scaling[74] = 0.05;
        cost_scaling[75] = 0.05;
        cost_scaling[76] = 0.05;
        cost_scaling[77] = 0.05;
        cost_scaling[78] = 0.05;
        cost_scaling[79] = 0.05;
        cost_scaling[80] = 0.05;
        cost_scaling[81] = 0.05;
        cost_scaling[82] = 0.05;
        cost_scaling[83] = 0.05;
        cost_scaling[84] = 0.05;
        cost_scaling[85] = 0.05;
        cost_scaling[86] = 0.05;
        cost_scaling[87] = 0.05;
        cost_scaling[88] = 0.05;
        cost_scaling[89] = 0.05;
        cost_scaling[90] = 0.05;
        cost_scaling[91] = 0.05;
        cost_scaling[92] = 0.05;
        cost_scaling[93] = 0.05;
        cost_scaling[94] = 0.05;
        cost_scaling[95] = 0.05;
        cost_scaling[96] = 0.05;
        cost_scaling[97] = 0.05;
        cost_scaling[98] = 0.05;
        cost_scaling[99] = 0.05;
        cost_scaling[100] = 0.05;
        cost_scaling[101] = 0.05;
        cost_scaling[102] = 0.05;
        cost_scaling[103] = 0.05;
        cost_scaling[104] = 0.05;
        cost_scaling[105] = 0.05;
        cost_scaling[106] = 0.05;
        cost_scaling[107] = 0.05;
        cost_scaling[108] = 0.05;
        cost_scaling[109] = 0.05;
        cost_scaling[110] = 0.05;
        cost_scaling[111] = 0.05;
        cost_scaling[112] = 0.05;
        cost_scaling[113] = 0.05;
        cost_scaling[114] = 0.05;
        cost_scaling[115] = 0.05;
        cost_scaling[116] = 0.05;
        cost_scaling[117] = 0.05;
        cost_scaling[118] = 0.05;
        cost_scaling[119] = 0.05;
        cost_scaling[120] = 0.05;
        cost_scaling[121] = 0.05;
        cost_scaling[122] = 0.05;
        cost_scaling[123] = 0.05;
        cost_scaling[124] = 0.05;
        cost_scaling[125] = 0.05;
        cost_scaling[126] = 0.05;
        cost_scaling[127] = 0.05;
        cost_scaling[128] = 0.05;
        cost_scaling[129] = 0.05;
        cost_scaling[130] = 0.05;
        cost_scaling[131] = 0.05;
        cost_scaling[132] = 0.05;
        cost_scaling[133] = 0.05;
        cost_scaling[134] = 0.05;
        cost_scaling[135] = 0.05;
        cost_scaling[136] = 0.05;
        cost_scaling[137] = 0.05;
        cost_scaling[138] = 0.05;
        cost_scaling[139] = 0.05;
        cost_scaling[140] = 0.05;
        cost_scaling[141] = 0.05;
        cost_scaling[142] = 0.05;
        cost_scaling[143] = 0.05;
        cost_scaling[144] = 0.05;
        cost_scaling[145] = 0.05;
        cost_scaling[146] = 0.05;
        cost_scaling[147] = 0.05;
        cost_scaling[148] = 0.05;
        cost_scaling[149] = 0.05;
        cost_scaling[150] = 0.05;
        cost_scaling[151] = 0.05;
        cost_scaling[152] = 0.05;
        cost_scaling[153] = 0.05;
        cost_scaling[154] = 0.05;
        cost_scaling[155] = 0.05;
        cost_scaling[156] = 0.05;
        cost_scaling[157] = 0.05;
        cost_scaling[158] = 0.05;
        cost_scaling[159] = 0.05;
        cost_scaling[160] = 0.05;
        cost_scaling[161] = 0.05;
        cost_scaling[162] = 0.05;
        cost_scaling[163] = 0.05;
        cost_scaling[164] = 0.05;
        cost_scaling[165] = 0.05;
        cost_scaling[166] = 0.05;
        cost_scaling[167] = 0.05;
        cost_scaling[168] = 0.05;
        cost_scaling[169] = 0.05;
        cost_scaling[170] = 0.05;
        cost_scaling[171] = 0.05;
        cost_scaling[172] = 0.05;
        cost_scaling[173] = 0.05;
        cost_scaling[174] = 0.05;
        cost_scaling[175] = 0.05;
        cost_scaling[176] = 0.05;
        cost_scaling[177] = 0.05;
        cost_scaling[178] = 0.05;
        cost_scaling[179] = 0.05;
        cost_scaling[180] = 0.05;
        cost_scaling[181] = 0.05;
        cost_scaling[182] = 0.05;
        cost_scaling[183] = 0.05;
        cost_scaling[184] = 0.05;
        cost_scaling[185] = 0.05;
        cost_scaling[186] = 0.05;
        cost_scaling[187] = 0.05;
        cost_scaling[188] = 0.05;
        cost_scaling[189] = 0.05;
        cost_scaling[190] = 0.05;
        cost_scaling[191] = 0.05;
        cost_scaling[192] = 0.05;
        cost_scaling[193] = 0.05;
        cost_scaling[194] = 0.05;
        cost_scaling[195] = 0.05;
        cost_scaling[196] = 0.05;
        cost_scaling[197] = 0.05;
        cost_scaling[198] = 0.05;
        cost_scaling[199] = 0.05;
        cost_scaling[200] = 0.05;
        cost_scaling[201] = 0.05;
        cost_scaling[202] = 0.05;
        cost_scaling[203] = 0.05;
        cost_scaling[204] = 0.05;
        cost_scaling[205] = 0.05;
        cost_scaling[206] = 0.05;
        cost_scaling[207] = 0.05;
        cost_scaling[208] = 0.05;
        cost_scaling[209] = 0.05;
        cost_scaling[210] = 0.05;
        cost_scaling[211] = 0.05;
        cost_scaling[212] = 0.05;
        cost_scaling[213] = 0.05;
        cost_scaling[214] = 0.05;
        cost_scaling[215] = 0.05;
        cost_scaling[216] = 0.05;
        cost_scaling[217] = 0.05;
        cost_scaling[218] = 0.05;
        cost_scaling[219] = 0.05;
        cost_scaling[220] = 0.05;
        cost_scaling[221] = 0.05;
        cost_scaling[222] = 0.05;
        cost_scaling[223] = 0.05;
        cost_scaling[224] = 0.05;
        cost_scaling[225] = 0.05;
        cost_scaling[226] = 0.05;
        cost_scaling[227] = 0.05;
        cost_scaling[228] = 0.05;
        cost_scaling[229] = 0.05;
        cost_scaling[230] = 0.05;
        cost_scaling[231] = 0.05;
        cost_scaling[232] = 0.05;
        cost_scaling[233] = 0.05;
        cost_scaling[234] = 0.05;
        cost_scaling[235] = 0.05;
        cost_scaling[236] = 0.05;
        cost_scaling[237] = 0.05;
        cost_scaling[238] = 0.05;
        cost_scaling[239] = 0.05;
        cost_scaling[240] = 0.05;
        cost_scaling[241] = 0.05;
        cost_scaling[242] = 0.05;
        cost_scaling[243] = 0.05;
        cost_scaling[244] = 0.05;
        cost_scaling[245] = 0.05;
        cost_scaling[246] = 0.05;
        cost_scaling[247] = 0.05;
        cost_scaling[248] = 0.05;
        cost_scaling[249] = 0.05;
        cost_scaling[250] = 0.05;
        cost_scaling[251] = 0.05;
        cost_scaling[252] = 0.05;
        cost_scaling[253] = 0.05;
        cost_scaling[254] = 0.05;
        cost_scaling[255] = 0.05;
        cost_scaling[256] = 0.05;
        cost_scaling[257] = 0.05;
        cost_scaling[258] = 0.05;
        cost_scaling[259] = 0.05;
        cost_scaling[260] = 0.05;
        cost_scaling[261] = 0.05;
        cost_scaling[262] = 0.05;
        cost_scaling[263] = 0.05;
        cost_scaling[264] = 0.05;
        cost_scaling[265] = 0.05;
        cost_scaling[266] = 0.05;
        cost_scaling[267] = 0.05;
        cost_scaling[268] = 0.05;
        cost_scaling[269] = 0.05;
        cost_scaling[270] = 0.05;
        cost_scaling[271] = 0.05;
        cost_scaling[272] = 0.05;
        cost_scaling[273] = 0.05;
        cost_scaling[274] = 0.05;
        cost_scaling[275] = 0.05;
        cost_scaling[276] = 0.05;
        cost_scaling[277] = 0.05;
        cost_scaling[278] = 0.05;
        cost_scaling[279] = 0.05;
        cost_scaling[280] = 0.05;
        cost_scaling[281] = 0.05;
        cost_scaling[282] = 0.05;
        cost_scaling[283] = 0.05;
        cost_scaling[284] = 0.05;
        cost_scaling[285] = 0.05;
        cost_scaling[286] = 0.05;
        cost_scaling[287] = 0.05;
        cost_scaling[288] = 0.05;
        cost_scaling[289] = 0.05;
        cost_scaling[290] = 0.05;
        cost_scaling[291] = 0.05;
        cost_scaling[292] = 0.05;
        cost_scaling[293] = 0.05;
        cost_scaling[294] = 0.05;
        cost_scaling[295] = 0.05;
        cost_scaling[296] = 0.05;
        cost_scaling[297] = 0.05;
        cost_scaling[298] = 0.05;
        cost_scaling[299] = 0.05;
        cost_scaling[300] = 0.05;
        cost_scaling[301] = 0.05;
        cost_scaling[302] = 0.05;
        cost_scaling[303] = 0.05;
        cost_scaling[304] = 0.05;
        cost_scaling[305] = 0.05;
        cost_scaling[306] = 0.05;
        cost_scaling[307] = 0.05;
        cost_scaling[308] = 0.05;
        cost_scaling[309] = 0.05;
        cost_scaling[310] = 0.05;
        cost_scaling[311] = 0.05;
        cost_scaling[312] = 0.05;
        cost_scaling[313] = 0.05;
        cost_scaling[314] = 0.05;
        cost_scaling[315] = 0.05;
        cost_scaling[316] = 0.05;
        cost_scaling[317] = 0.05;
        cost_scaling[318] = 0.05;
        cost_scaling[319] = 0.05;
        cost_scaling[320] = 0.05;
        cost_scaling[321] = 0.05;
        cost_scaling[322] = 0.05;
        cost_scaling[323] = 0.05;
        cost_scaling[324] = 0.05;
        cost_scaling[325] = 0.05;
        cost_scaling[326] = 0.05;
        cost_scaling[327] = 0.05;
        cost_scaling[328] = 0.05;
        cost_scaling[329] = 0.05;
        cost_scaling[330] = 0.05;
        cost_scaling[331] = 0.05;
        cost_scaling[332] = 0.05;
        cost_scaling[333] = 0.05;
        cost_scaling[334] = 0.05;
        cost_scaling[335] = 0.05;
        cost_scaling[336] = 0.05;
        cost_scaling[337] = 0.05;
        cost_scaling[338] = 0.05;
        cost_scaling[339] = 0.05;
        cost_scaling[340] = 0.05;
        cost_scaling[341] = 0.05;
        cost_scaling[342] = 0.05;
        cost_scaling[343] = 0.05;
        cost_scaling[344] = 0.05;
        cost_scaling[345] = 0.05;
        cost_scaling[346] = 0.05;
        cost_scaling[347] = 0.05;
        cost_scaling[348] = 0.05;
        cost_scaling[349] = 0.05;
        cost_scaling[350] = 0.05;
        cost_scaling[351] = 0.05;
        cost_scaling[352] = 0.05;
        cost_scaling[353] = 0.05;
        cost_scaling[354] = 0.05;
        cost_scaling[355] = 0.05;
        cost_scaling[356] = 0.05;
        cost_scaling[357] = 0.05;
        cost_scaling[358] = 0.05;
        cost_scaling[359] = 0.05;
        cost_scaling[360] = 0.05;
        cost_scaling[361] = 0.05;
        cost_scaling[362] = 0.05;
        cost_scaling[363] = 0.05;
        cost_scaling[364] = 0.05;
        cost_scaling[365] = 0.05;
        cost_scaling[366] = 0.05;
        cost_scaling[367] = 0.05;
        cost_scaling[368] = 0.05;
        cost_scaling[369] = 0.05;
        cost_scaling[370] = 0.05;
        cost_scaling[371] = 0.05;
        cost_scaling[372] = 0.05;
        cost_scaling[373] = 0.05;
        cost_scaling[374] = 0.05;
        cost_scaling[375] = 0.05;
        cost_scaling[376] = 0.05;
        cost_scaling[377] = 0.05;
        cost_scaling[378] = 0.05;
        cost_scaling[379] = 0.05;
        cost_scaling[380] = 0.05;
        cost_scaling[381] = 0.05;
        cost_scaling[382] = 0.05;
        cost_scaling[383] = 0.05;
        cost_scaling[384] = 0.05;
        cost_scaling[385] = 0.05;
        cost_scaling[386] = 0.05;
        cost_scaling[387] = 0.05;
        cost_scaling[388] = 0.05;
        cost_scaling[389] = 0.05;
        cost_scaling[390] = 0.05;
        cost_scaling[391] = 0.05;
        cost_scaling[392] = 0.05;
        cost_scaling[393] = 0.05;
        cost_scaling[394] = 0.05;
        cost_scaling[395] = 0.05;
        cost_scaling[396] = 0.05;
        cost_scaling[397] = 0.05;
        cost_scaling[398] = 0.05;
        cost_scaling[399] = 0.05;
        cost_scaling[400] = 0.05;
        cost_scaling[401] = 0.05;
        cost_scaling[402] = 0.05;
        cost_scaling[403] = 0.05;
        cost_scaling[404] = 0.05;
        cost_scaling[405] = 0.05;
        cost_scaling[406] = 0.05;
        cost_scaling[407] = 0.05;
        cost_scaling[408] = 0.05;
        cost_scaling[409] = 0.05;
        cost_scaling[410] = 0.05;
        cost_scaling[411] = 0.05;
        cost_scaling[412] = 0.05;
        cost_scaling[413] = 0.05;
        cost_scaling[414] = 0.05;
        cost_scaling[415] = 0.05;
        cost_scaling[416] = 0.05;
        cost_scaling[417] = 0.05;
        cost_scaling[418] = 0.05;
        cost_scaling[419] = 0.05;
        cost_scaling[420] = 0.05;
        cost_scaling[421] = 0.05;
        cost_scaling[422] = 0.05;
        cost_scaling[423] = 0.05;
        cost_scaling[424] = 0.05;
        cost_scaling[425] = 0.05;
        cost_scaling[426] = 0.05;
        cost_scaling[427] = 0.05;
        cost_scaling[428] = 0.05;
        cost_scaling[429] = 0.05;
        cost_scaling[430] = 0.05;
        cost_scaling[431] = 0.05;
        cost_scaling[432] = 0.05;
        cost_scaling[433] = 0.05;
        cost_scaling[434] = 0.05;
        cost_scaling[435] = 0.05;
        cost_scaling[436] = 0.05;
        cost_scaling[437] = 0.05;
        cost_scaling[438] = 0.05;
        cost_scaling[439] = 0.05;
        cost_scaling[440] = 0.05;
        cost_scaling[441] = 0.05;
        cost_scaling[442] = 0.05;
        cost_scaling[443] = 0.05;
        cost_scaling[444] = 0.05;
        cost_scaling[445] = 0.05;
        cost_scaling[446] = 0.05;
        cost_scaling[447] = 0.05;
        cost_scaling[448] = 0.05;
        cost_scaling[449] = 0.05;
        cost_scaling[450] = 0.05;
        cost_scaling[451] = 0.05;
        cost_scaling[452] = 0.05;
        cost_scaling[453] = 0.05;
        cost_scaling[454] = 0.05;
        cost_scaling[455] = 0.05;
        cost_scaling[456] = 0.05;
        cost_scaling[457] = 0.05;
        cost_scaling[458] = 0.05;
        cost_scaling[459] = 0.05;
        cost_scaling[460] = 0.05;
        cost_scaling[461] = 0.05;
        cost_scaling[462] = 0.05;
        cost_scaling[463] = 0.05;
        cost_scaling[464] = 0.05;
        cost_scaling[465] = 0.05;
        cost_scaling[466] = 0.05;
        cost_scaling[467] = 0.05;
        cost_scaling[468] = 0.05;
        cost_scaling[469] = 0.05;
        cost_scaling[470] = 0.05;
        cost_scaling[471] = 0.05;
        cost_scaling[472] = 0.05;
        cost_scaling[473] = 0.05;
        cost_scaling[474] = 0.05;
        cost_scaling[475] = 0.05;
        cost_scaling[476] = 0.05;
        cost_scaling[477] = 0.05;
        cost_scaling[478] = 0.05;
        cost_scaling[479] = 0.05;
        cost_scaling[480] = 0.05;
        cost_scaling[481] = 0.05;
        cost_scaling[482] = 0.05;
        cost_scaling[483] = 0.05;
        cost_scaling[484] = 0.05;
        cost_scaling[485] = 0.05;
        cost_scaling[486] = 0.05;
        cost_scaling[487] = 0.05;
        cost_scaling[488] = 0.05;
        cost_scaling[489] = 0.05;
        cost_scaling[490] = 0.05;
        cost_scaling[491] = 0.05;
        cost_scaling[492] = 0.05;
        cost_scaling[493] = 0.05;
        cost_scaling[494] = 0.05;
        cost_scaling[495] = 0.05;
        cost_scaling[496] = 0.05;
        cost_scaling[497] = 0.05;
        cost_scaling[498] = 0.05;
        cost_scaling[499] = 0.05;
        cost_scaling[500] = 0.05;
        cost_scaling[501] = 0.05;
        cost_scaling[502] = 0.05;
        cost_scaling[503] = 0.05;
        cost_scaling[504] = 0.05;
        cost_scaling[505] = 0.05;
        cost_scaling[506] = 0.05;
        cost_scaling[507] = 0.05;
        cost_scaling[508] = 0.05;
        cost_scaling[509] = 0.05;
        cost_scaling[510] = 0.05;
        cost_scaling[511] = 0.05;
        cost_scaling[512] = 0.05;
        cost_scaling[513] = 0.05;
        cost_scaling[514] = 0.05;
        cost_scaling[515] = 0.05;
        cost_scaling[516] = 0.05;
        cost_scaling[517] = 0.05;
        cost_scaling[518] = 0.05;
        cost_scaling[519] = 0.05;
        cost_scaling[520] = 0.05;
        cost_scaling[521] = 0.05;
        cost_scaling[522] = 0.05;
        cost_scaling[523] = 0.05;
        cost_scaling[524] = 0.05;
        cost_scaling[525] = 0.05;
        cost_scaling[526] = 0.05;
        cost_scaling[527] = 0.05;
        cost_scaling[528] = 0.05;
        cost_scaling[529] = 0.05;
        cost_scaling[530] = 0.05;
        cost_scaling[531] = 0.05;
        cost_scaling[532] = 0.05;
        cost_scaling[533] = 0.05;
        cost_scaling[534] = 0.05;
        cost_scaling[535] = 0.05;
        cost_scaling[536] = 0.05;
        cost_scaling[537] = 0.05;
        cost_scaling[538] = 0.05;
        cost_scaling[539] = 0.05;
        cost_scaling[540] = 0.05;
        cost_scaling[541] = 0.05;
        cost_scaling[542] = 0.05;
        cost_scaling[543] = 0.05;
        cost_scaling[544] = 0.05;
        cost_scaling[545] = 0.05;
        cost_scaling[546] = 0.05;
        cost_scaling[547] = 0.05;
        cost_scaling[548] = 0.05;
        cost_scaling[549] = 0.05;
        cost_scaling[550] = 0.05;
        cost_scaling[551] = 0.05;
        cost_scaling[552] = 0.05;
        cost_scaling[553] = 0.05;
        cost_scaling[554] = 0.05;
        cost_scaling[555] = 0.05;
        cost_scaling[556] = 0.05;
        cost_scaling[557] = 0.05;
        cost_scaling[558] = 0.05;
        cost_scaling[559] = 0.05;
        cost_scaling[560] = 0.05;
        cost_scaling[561] = 0.05;
        cost_scaling[562] = 0.05;
        cost_scaling[563] = 0.05;
        cost_scaling[564] = 0.05;
        cost_scaling[565] = 0.05;
        cost_scaling[566] = 0.05;
        cost_scaling[567] = 0.05;
        cost_scaling[568] = 0.05;
        cost_scaling[569] = 0.05;
        cost_scaling[570] = 0.05;
        cost_scaling[571] = 0.05;
        cost_scaling[572] = 0.05;
        cost_scaling[573] = 0.05;
        cost_scaling[574] = 0.05;
        cost_scaling[575] = 0.05;
        cost_scaling[576] = 0.05;
        cost_scaling[577] = 0.05;
        cost_scaling[578] = 0.05;
        cost_scaling[579] = 0.05;
        cost_scaling[580] = 0.05;
        cost_scaling[581] = 0.05;
        cost_scaling[582] = 0.05;
        cost_scaling[583] = 0.05;
        cost_scaling[584] = 0.05;
        cost_scaling[585] = 0.05;
        cost_scaling[586] = 0.05;
        cost_scaling[587] = 0.05;
        cost_scaling[588] = 0.05;
        cost_scaling[589] = 0.05;
        cost_scaling[590] = 0.05;
        cost_scaling[591] = 0.05;
        cost_scaling[592] = 0.05;
        cost_scaling[593] = 0.05;
        cost_scaling[594] = 0.05;
        cost_scaling[595] = 0.05;
        cost_scaling[596] = 0.05;
        cost_scaling[597] = 0.05;
        cost_scaling[598] = 0.05;
        cost_scaling[599] = 0.05;
        cost_scaling[600] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }


    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "impl_dae_fun", &capsule->impl_dae_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_fun_jac_x_xdot_z", &capsule->impl_dae_fun_jac_x_xdot_z[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_jac_x_xdot_u", &capsule->impl_dae_jac_x_xdot_u_z[i]);
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 1;
    W_0[2+(NY0) * 2] = 0.000001;
    W_0[3+(NY0) * 3] = 1;
    W_0[4+(NY0) * 4] = 0.000001;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* Vx_0 = calloc(NY0*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_0[0+(NY0) * 0] = 1;
    Vx_0[1+(NY0) * 1] = 1;
    Vx_0[2+(NY0) * 2] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);
    double* Vu_0 = calloc(NY0*NU, sizeof(double));
    // change only the non-zero elements:
    Vu_0[3+(NY0) * 0] = 1;
    Vu_0[4+(NY0) * 1] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 1;
    W[2+(NY) * 2] = 0.000001;
    W[3+(NY) * 3] = 1;
    W[4+(NY) * 4] = 0.000001;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* Vx = calloc(NY*NX, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(NY) * 0] = 1;
    Vx[1+(NY) * 1] = 1;
    Vx[2+(NY) * 2] = 1;
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    double* Vu = calloc(NY*NU, sizeof(double));
    // change only the non-zero elements:
    Vu[3+(NY) * 0] = 1;
    Vu[4+(NY) * 1] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 20.241427671158934;
    W_e[0+(NYN) * 1] = 0.005979345913602564;
    W_e[0+(NYN) * 2] = -0.021262470676863102;
    W_e[1+(NYN) * 0] = 0.005979345913602564;
    W_e[1+(NYN) * 1] = 0.004124781838203661;
    W_e[1+(NYN) * 2] = -0.000010887892379631266;
    W_e[2+(NYN) * 0] = -0.021262470676863102;
    W_e[2+(NYN) * 1] = -0.000010887892379631266;
    W_e[2+(NYN) * 2] = 0.000052834419134583414;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    double* Vx_e = calloc(NYN*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_e[0+(NYN) * 0] = 1;
    Vx_e[1+(NYN) * 1] = 1;
    Vx_e[2+(NYN) * 2] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    free(Vx_e);






    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:
    lbx0[0] = 0.000001;
    ubx0[0] = 0.000001;
    lbx0[1] = 0.000001;
    ubx0[1] = 0.000001;
    lbx0[2] = 0.000001;
    ubx0[2] = 0.000001;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(3 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);








    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -8;
    ubu[0] = 8;
    lbu[1] = -15;
    ubu[1] = 15;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);








    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 0;
    idxbx[1] = 1;
    idxbx[2] = 2;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    lbx[0] = -1.267;
    ubx[0] = 1.267;
    ubx[1] = 90;
    lbx[2] = -47.40291;
    ubx[2] = 47.40291;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);







    /* terminal constraints */











    // set up general constraints for last stage
    double* C_e = calloc(NGN*NX, sizeof(double));
    double* lug_e = calloc(2*NGN, sizeof(double));
    double* lg_e = lug_e;
    double* ug_e = lug_e + NGN;
    C_e[0+NGN * 0] = 1.134395425881501;
    C_e[0+NGN * 1] = 0.001872741751232452;
    C_e[0+NGN * 2] = -0.00925903328071332;
    C_e[1+NGN * 0] = 28.26937386101787;
    C_e[1+NGN * 1] = 0.1686103927669081;
    C_e[1+NGN * 2] = 0.1859027881191664;
    C_e[2+NGN * 0] = -1.134395425881501;
    C_e[2+NGN * 1] = -0.001872741751232452;
    C_e[2+NGN * 2] = 0.00925903328071332;
    C_e[3+NGN * 0] = 1.3438059523931931;
    C_e[3+NGN * 1] = -0.9784341783792972;
    C_e[3+NGN * 2] = -0.03469492756386506;
    C_e[4+NGN * 0] = 1.1345833147085147;
    C_e[4+NGN * 1] = 0.0018619875730820589;
    C_e[4+NGN * 2] = -0.009203824335094136;
    C_e[5+NGN * 0] = 28.100993986213776;
    C_e[5+NGN * 1] = 0.1674251810775851;
    C_e[5+NGN * 2] = 0.19173521068061233;
    C_e[6+NGN * 0] = 1.3351881127103302;
    C_e[6+NGN * 1] = -0.978628540372808;
    C_e[6+NGN * 2] = -0.034441038746133544;
    C_e[7+NGN * 0] = 1.1347471338757562;
    C_e[7+NGN * 1] = 0.001851100051546991;
    C_e[7+NGN * 2] = -0.009147986727748024;
    C_e[8+NGN * 0] = 27.930691967165274;
    C_e[8+NGN * 1] = 0.1662312200806271;
    C_e[8+NGN * 2] = 0.19760812939986125;
    C_e[9+NGN * 0] = 1.3264881135438111;
    C_e[9+NGN * 1] = -0.978823280909674;
    C_e[9+NGN * 2] = -0.03418549977830552;
    C_e[10+NGN * 0] = 1.1348866956437478;
    C_e[10+NGN * 1] = 0.0018400784104704527;
    C_e[10+NGN * 2] = -0.00909151690704818;
    C_e[11+NGN * 0] = 27.75845693462176;
    C_e[11+NGN * 1] = 0.165028470302586;
    C_e[11+NGN * 2] = 0.2035217219142388;
    C_e[12+NGN * 0] = 1.3177055242324986;
    C_e[12+NGN * 1] = -0.9790184014946643;
    C_e[12+NGN * 2] = -0.033928303940502266;
    C_e[13+NGN * 0] = 1.1350018113828564;
    C_e[13+NGN * 1] = 0.001828921870776161;
    C_e[13+NGN * 2] = -0.009034411308425198;
    C_e[14+NGN * 0] = 27.584277979683417;
    C_e[14+NGN * 1] = 0.16381689216587692;
    C_e[14+NGN * 2] = 0.2094761663048376;
    C_e[15+NGN * 0] = 1.3088399126794843;
    C_e[15+NGN * 1] = -0.979213903635938;
    C_e[15+NGN * 2] = -0.033669444498737836;
    C_e[16+NGN * 0] = 1.1350922915707289;
    C_e[16+NGN * 1] = 0.0018176296504635384;
    C_e[16+NGN * 2] = -0.00897666635434827;
    C_e[17+NGN * 0] = 27.408144153743304;
    C_e[17+NGN * 1] = 0.16259644598887169;
    C_e[17+NGN * 2] = 0.21547164109590644;
    C_e[18+NGN * 0] = 1.2998908453508256;
    C_e[18+NGN * 1] = -0.9794097888450328;
    C_e[18+NGN * 2] = -0.03340891470496259;
    C_e[19+NGN * 0] = 1.135157945789727;
    C_e[19+NGN * 1] = 0.0018062009646029353;
    C_e[19+NGN * 2] = -0.008918278454306506;
    C_e[20+NGN * 0] = 27.230044468429956;
    C_e[20+NGN * 1] = 0.16136709198599555;
    C_e[20+NGN * 2] = 0.2215083252542245;
    C_e[21+NGN * 0] = 1.2908578872743044;
    C_e[21+NGN * 1] = -0.9796060586368608;
    C_e[21+NGN * 2] = -0.03314670779710741;
    C_e[22+NGN * 0] = 1.1351985827243631;
    C_e[22+NGN * 1] = 0.001794635025330882;
    C_e[22+NGN * 2] = -0.008859244004790412;
    C_e[23+NGN * 0] = 27.049967895550346;
    C_e[23+NGN * 1] = 0.16012879026782703;
    C_e[23+NGN * 2] = 0.22758639818846116;
    C_e[24+NGN * 0] = 1.2817406020382116;
    C_e[24+NGN * 1] = -0.9798027145296964;
    C_e[24+NGN * 2] = -0.032882816999128546;
    C_e[25+NGN * 0] = 1.135214010158737;
    C_e[25+NGN * 1] = 0.0017829310418453715;
    C_e[25+NGN * 2] = -0.008799559389273532;
    C_e[26+NGN * 0] = 26.867903367033396;
    C_e[26+NGN * 1] = 0.15888150084120073;
    C_e[26+NGN * 2] = 0.23370603974852075;
    C_e[27+NGN * 0] = 1.2725385517901515;
    C_e[27+NGN * 1] = -0.979999758045171;
    C_e[27+NGN * 2] = -0.03261723552105312;
    C_e[28+NGN * 0] = 1.1352040349739705;
    C_e[28+NGN * 1] = 0.00177108822040117;
    C_e[28+NGN * 2] = -0.008739220978194226;
    C_e[29+NGN * 0] = 26.6838397748739;
    C_e[29+NGN * 1] = 0.15762518360931335;
    C_e[29+NGN * 2] = 0.23986743022487247;
    C_e[30+NGN * 0] = 1.2632512972358727;
    C_e[30+NGN * 1] = -0.9801971907082632;
    C_e[30+NGN * 2] = -0.0323499565590253;
    C_e[31+NGN * 0] = 1.135168463145649;
    C_e[31+NGN * 1] = 0.0017591057643051586;
    C_e[31+NGN * 2] = -0.008678225128937603;
    C_e[32+NGN * 0] = 26.497765971076937;
    C_e[32+NGN * 1] = 0.1563597983718328;
    C_e[32+NGN * 2] = 0.24607075034786532;
    C_e[33+NGN * 0] = 1.2538783976381171;
    C_e[33+NGN * 1] = -0.98039501404729;
    C_e[33+NGN * 2] = -0.032080973295353137;
    C_e[34+NGN * 0] = 1.135107099741256;
    C_e[34+NGN * 1] = 0.0017469828739117071;
    C_e[34+NGN * 2] = -0.008616568185817622;
    C_e[35+NGN * 0] = 26.309670767602736;
    C_e[35+NGN * 1] = 0.1550853048250106;
    C_e[35+NGN * 2] = 0.25231618128702776;
    C_e[36+NGN * 0] = 1.2444194108154996;
    C_e[36+NGN * 1] = -0.9805932295938982;
    C_e[36+NGN * 2] = -0.031810278898556074;
    C_e[37+NGN * 0] = 1.135019748917615;
    C_e[37+NGN * 1] = 0.0017347187466180765;
    C_e[37+NGN * 2] = -0.008554246480059334;
    C_e[38+NGN * 0] = 26.119542936312076;
    C_e[38+NGN * 1] = 0.15380166256179714;
    C_e[38+NGN * 2] = 0.2586039046503523;
    C_e[39+NGN * 0] = 1.2348738931414058;
    C_e[39+NGN * 1] = -0.980791838883056;
    C_e[39+NGN * 2] = -0.031537866523413116;
    C_e[40+NGN * 0] = 1.134906213918328;
    C_e[40+NGN * 1] = 0.001722312576859854;
    C_e[40+NGN * 2] = -0.008491256329781297;
    C_e[41+NGN * 0] = 25.927371208912085;
    C_e[41+NGN * 1] = 0.15250883107196067;
    C_e[41+NGN * 2] = 0.2649341024835647;
    C_e[42+NGN * 0] = 1.2252413995429177;
    C_e[42+NGN * 1] = -0.9809908434530434;
    C_e[42+NGN * 2] = -0.03126372931101169;
    C_e[43+NGN * 0] = 1.1347662970712171;
    C_e[43+NGN * 1] = 0.0017097635561064209;
    C_e[43+NGN * 2] = -0.008427594039978139;
    C_e[44+NGN * 0] = 25.733144276902603;
    C_e[44+NGN * 1] = 0.15120676974220895;
    C_e[44+NGN * 2] = 0.27130695726937754;
    C_e[45+NGN * 0] = 1.2155214834997603;
    C_e[45+NGN * 1] = -0.9811902448454431;
    C_e[45+NGN * 2] = -0.030987860388797184;
    C_e[46+NGN * 0] = 1.1345997997857669;
    C_e[46+NGN * 1] = 0.0016970708728564505;
    C_e[46+NGN * 2] = -0.00836325590250329;
    C_e[47+NGN * 0] = 25.536850791523;
    C_e[47+NGN * 1] = 0.14989543785631454;
    C_e[47+NGN * 2] = 0.2777226519267287;
    C_e[48+NGN * 0] = 1.2057136970432774;
    C_e[48+NGN * 1] = -0.9813900446051316;
    C_e[48+NGN * 2] = -0.030710252870623157;
    C_e[49+NGN * 0] = 1.1344065225505668;
    C_e[49+NGN * 1] = 0.001684233712633438;
    C_e[49+NGN * 2] = -0.00829823819605188;
    C_e[50+NGN * 0] = 25.338479363699506;
    C_e[50+NGN * 1] = 0.14857479459524328;
    C_e[50+NGN * 2] = 0.2841813698100037;
    C_e[51+NGN * 0] = 1.1958175907554252;
    C_e[51+NGN * 1] = -0.9815902442802684;
    C_e[51+NGN * 2] = -0.03043089985680227;
    C_e[52+NGN * 0] = 1.1341862649307566;
    C_e[52+NGN * 1] = 0.0016712512579812662;
    C_e[52+NGN * 2] = -0.008232537186143791;
    C_e[53+NGN * 0] = 25.13801856399301;
    C_e[53+NGN * 1] = 0.1472447990372859;
    C_e[53+NGN * 2] = 0.2906832947082428;
    C_e[54+NGN * 0] = 1.1858327137677986;
    C_e[54+NGN * 1] = -0.981790845422288;
    C_e[54+NGN * 2] = -0.030149794434157873;
    C_e[55+NGN * 0] = 1.1339388255654708;
    C_e[55+NGN * 1] = 0.001658122688459802;
    C_e[55+NGN * 2] = -0.008166149125106884;
    C_e[56+NGN * 0] = 24.935456922547456;
    C_e[56+NGN * 1] = 0.14590541015819317;
    C_e[56+NGN * 2] = 0.2972286108443317;
    C_e[57+NGN * 0] = 1.1757586137606768;
    C_e[57+NGN * 1] = -0.9819918495858893;
    C_e[57+NGN * 2] = -0.02986692967607633;
    C_e[58+NGN * 0] = 1.1336640021652866;
    C_e[58+NGN * 1] = 0.0016448471806405252;
    C_e[58+NGN * 2] = -0.008099070252060397;
    C_e[59+NGN * 0] = 24.730782929038643;
    C_e[59+NGN * 1] = 0.14455658683131423;
    C_e[59+NGN * 2] = 0.303817502874177;
    C_e[60+NGN * 0] = 1.1655948369620968;
    C_e[60+NGN * 1] = -0.9821932583290254;
    C_e[60+NGN * 2] = -0.029582298642560005;
    C_e[61+NGN * 0] = 1.1333615915096709;
    C_e[61+NGN * 1] = 0.0016314239081021945;
    C_e[61+NGN * 2] = -0.008031296792898504;
    C_e[62+NGN * 0] = 24.523985032623624;
    C_e[62+NGN * 1] = 0.14319828782773836;
    C_e[62+NGN * 2] = 0.3104501558858649;
    C_e[63+NGN * 0] = 1.155340928146952;
    C_e[63+NGN * 1] = -0.9823950732128942;
    C_e[63+NGN * 2] = -0.029295894380280987;
    C_e[64+NGN * 0] = 1.1330313894444306;
    C_e[64+NGN * 1] = 0.0016178520414265442;
    C_e[64+NGN * 2] = -0.007962824960274053;
    C_e[65+NGN * 0] = 24.31505164189059;
    C_e[65+NGN * 1] = 0.1418304718164401;
    C_e[65+NGN * 2] = 0.3171267553988041;
    C_e[66+NGN * 0] = 1.144996430636117;
    C_e[66+NGN * 1] = -0.9825972958019278;
    C_e[66+NGN * 2] = -0.029007709922635502;
    C_e[67+NGN * 0] = 1.1326731908791643;
    C_e[67+NGN * 1] = 0.001604130748194018;
    C_e[67+NGN * 2] = -0.007893650953582475;
    C_e[68+NGN * 0] = 24.103971124809306;
    C_e[68+NGN * 1] = 0.14045309736442782;
    C_e[68+NGN * 2] = 0.32384748736285257;
    C_e[69+NGN * 0] = 1.1345608862955974;
    C_e[69+NGN * 1] = -0.982799927663782;
    C_e[69+NGN * 2] = -0.028717738289799077;
    C_e[70+NGN * 0] = 1.1322867897847135;
    C_e[70+NGN * 1] = 0.0015902591929795368;
    C_e[70+NGN * 2] = -0.007823770958945868;
    C_e[71+NGN * 0] = 23.89073180868203;
    C_e[71+NGN * 1] = 0.13906612293689563;
    C_e[71+NGN * 2] = 0.33061253815742775;
    C_e[72+NGN * 0] = 1.1240338355357065;
    C_e[72+NGN * 1] = -0.9830029703693258;
    C_e[72+NGN * 2] = -0.02842597248878235;
    C_e[73+NGN * 0] = 1.131871979190618;
    C_e[73+NGN * 1] = 0.0015762365373482995;
    C_e[73+NGN * 2] = -0.0077531811491972575;
    C_e[74+NGN * 0] = 23.675321980095077;
    C_e[74+NGN * 1] = 0.13766950689737883;
    C_e[74+NGN * 2] = 0.33742209459060035;
    C_e[75+NGN * 0] = 1.1134148173102691;
    C_e[75+NGN * 1] = -0.9832064254926312;
    C_e[75+NGN * 2] = -0.028132405513487693;
    C_e[76+NGN * 0] = 1.1314285511825717;
    C_e[76+NGN * 1] = 0.0015620619398516238;
    C_e[76+NGN * 2] = -0.007681877683865033;
    C_e[77+NGN * 0] = 23.4577298848708;
    C_e[77+NGN * 1] = 0.13626320750791282;
    C_e[77+NGN * 2] = 0.3442763438981716;
    C_e[78+NGN * 0] = 1.10270336911585;
    C_e[78+NGN * 1] = -0.9834102946109616;
    C_e[78+NGN * 2] = -0.027837030344766495;
    C_e[79+NGN * 0] = 1.13095629689988;
    C_e[79+NGN * 1] = 0.0015477345560228194;
    C_e[79+NGN * 2] = -0.007609856709157573;
    C_e[80+NGN * 0] = 23.237943728020205;
    C_e[80+NGN * 1] = 0.1348471829291954;
    C_e[80+NGN * 2] = 0.35117547374273367;
    C_e[81+NGN * 0] = 1.0918990269910116;
    C_e[81+NGN * 1] = -0.9836145793047614;
    C_e[81+NGN * 2] = -0.027539839950477214;
    C_e[82+NGN * 0] = 1.1304550065329204;
    C_e[82+NGN * 1] = 0.001533253538373098;
    C_e[82+NGN * 2] = -0.007537114357948041;
    C_e[83+NGN * 0] = 23.015951673696115;
    C_e[83+NGN * 1] = 0.13342139122075283;
    C_e[83+NGN * 2] = 0.3581196722127137;
    C_e[84+NGN * 0] = 1.0810013255155966;
    C_e[84+NGN * 1] = -0.9838192811576444;
    C_e[84+NGN * 2] = -0.02724082728554414;
    C_e[85+NGN * 0] = 1.1299244693206023;
    C_e[85+NGN * 1] = 0.0015186180363875223;
    C_e[85+NGN * 2] = -0.007463646749759374;
    C_e[86+NGN * 0] = 22.791741845146852;
    C_e[86+NGN * 1] = 0.13198579034110913;
    C_e[86+NGN * 2] = 0.3651091278214004;
    C_e[87+NGN * 0] = 1.0700097978100398;
    C_e[87+NGN * 1] = -0.9840244017563828;
    C_e[87+NGN * 2] = -0.026939985292016917;
    C_e[88+NGN * 0] = 1.1293644735478332;
    C_e[88+NGN * 1] = 0.0015038271965209888;
    C_e[88+NGN * 2] = -0.007389449990749452;
    C_e[89+NGN * 0] = 22.56530232467052;
    C_e[89+NGN * 1] = 0.13054033814795918;
    C_e[89+NGN * 2] = 0.3721440295059542;
    C_e[90+NGN * 0] = 1.0589239755347069;
    C_e[90+NGN * 1] = -0.9842299426908958;
    C_e[90+NGN * 2] = -0.026637306899130804;
    C_e[91+NGN * 0] = 1.1287748065429803;
    C_e[91+NGN * 1] = 0.0014888801621942502;
    C_e[91+NGN * 2] = -0.007314520173696454;
    C_e[92+NGN * 0] = 22.33662115356985;
    C_e[92+NGN * 1] = 0.1290849923983453;
    C_e[92+NGN * 2] = 0.3792245666263999;
    C_e[93+NGN * 0] = 1.0477433888892598;
    C_e[93+NGN * 1] = -0.9844359055542375;
    C_e[93+NGN * 2] = -0.026332785023367677;
    C_e[94+NGN * 0] = 1.1281552546753426;
    C_e[94+NGN * 1] = 0.0014737760737899752;
    C_e[94+NGN * 2] = -0.007238853377984404;
    C_e[95+NGN * 0] = 22.105686332107627;
    C_e[95+NGN * 1] = 0.12761971074883754;
    C_e[95+NGN * 2] = 0.3863509289646024;
    C_e[96+NGN * 0] = 1.0364675666120515;
    C_e[96+NGN * 1] = -0.984642291942586;
    C_e[96+NGN * 2] = -0.0260264125685178;
    C_e[97+NGN * 0] = 1.1275056033526176;
    C_e[97+NGN * 1] = 0.0014585140686488463;
    C_e[97+NGN * 2] = -0.007162445669588913;
    C_e[98+NGN * 0] = 21.87248581946268;
    C_e[98+NGN * 1] = 0.12614445075571742;
    C_e[98+NGN * 2] = 0.3935233067232246;
    C_e[99+NGN * 0] = 1.0250960359795491;
    C_e[99+NGN * 1] = -0.98484910345523;
    C_e[99+NGN * 2] = -0.02571818242574232;
    C_e[100+NGN * 0] = 1.126825637018375;
    C_e[100+NGN * 1] = 0.0014430932810656969;
    C_e[100+NGN * 2] = -0.007085293101063102;
    C_e[101+NGN * 0] = 21.63700753368648;
    C_e[101+NGN * 1] = 0.12465916987516557;
    C_e[101+NGN * 2] = 0.4007418905246687;
    C_e[102+NGN * 0] = 1.0136283228057834;
    C_e[102+NGN * 1] = -0.9850563416945582;
    C_e[102+NGN * 2] = -0.0254080874736366;
    C_e[103+NGN * 0] = 1.1261151391495303;
    C_e[103+NGN * 1] = 0.0014275128422856832;
    C_e[103+NGN * 2] = -0.007007391711523729;
    C_e[104+NGN * 0] = 21.399239351660334;
    C_e[104+NGN * 1] = 0.12316382546345286;
    C_e[104+NGN * 2] = 0.40800687140999914;
    C_e[105+NGN * 0] = 1.0020639514418306;
    C_e[105+NGN * 1] = -0.9852640082660464;
    C_e[105+NGN * 2] = -0.0250961205782942;
    C_e[106+NGN * 0] = 1.1253738922538237;
    C_e[106+NGN * 1] = 0.0014117718805005037;
    C_e[106+NGN * 2] = -0.006928737526637508;
    C_e[107+NGN * 0] = 21.159169109053146;
    C_e[107+NGN * 1] = 0.12165837477713536;
    C_e[107+NGN * 2] = 0.4153184408378485;
    C_e[108+NGN * 0] = 0.9904024447753188;
    C_e[108+NGN * 1] = -0.9854721047782448;
    C_e[108+NGN * 2] = -0.02478227459337175;
    C_e[109+NGN * 0] = 1.124601677867296;
    C_e[109+NGN * 1] = 0.0013958695208446505;
    C_e[109+NGN * 2] = -0.006849326558607627;
    C_e[110+NGN * 0] = 20.916784600279815;
    C_e[110+NGN * 1] = 0.120142774973253;
    C_e[110+NGN * 2] = 0.42267679068330527;
    C_e[111+NGN * 0] = 0.9786433242299672;
    C_e[111+NGN * 1] = -0.9856806328427656;
    C_e[111+NGN * 2] = -0.02446654236015453;
    C_e[112+NGN * 0] = 1.1237982765517742;
    C_e[112+NGN * 1] = 0.0013798048853917045;
    C_e[112+NGN * 2] = -0.006769154806160466;
    C_e[113+NGN * 0] = 20.672073578460225;
    C_e[113+NGN * 1] = 0.11861698310953184;
    C_e[113+NGN * 2] = 0.4300821132367838;
    C_e[114+NGN * 0] = 0.9667861097651532;
    C_e[114+NGN * 1] = -0.9858895940742708;
    C_e[114+NGN * 2] = -0.02414891670762281;
    C_e[115+NGN * 0] = 1.1229634678923546;
    C_e[115+NGN * 1] = 0.0013635770931506698;
    C_e[115+NGN * 2] = -0.0066882182545325115;
    C_e[116+NGN * 0] = 20.42502375537887;
    C_e[116+NGN * 1] = 0.11708095614459048;
    C_e[116+NGN * 2] = 0.4375346012028765;
    C_e[117+NGN * 0] = 0.9548303198755086;
    C_e[117+NGN * 1] = -0.9860989900904584;
    C_e[117+NGN * 2] = -0.023829390452519063;
    C_e[118+NGN * 0] = 1.1220970304948898;
    C_e[118+NGN * 1] = 0.001347185260062351;
    C_e[118+NGN * 2] = -0.006606512875457484;
    C_e[119+NGN * 0] = 20.175622801445055;
    C_e[119+NGN * 1] = 0.11553465093814988;
    C_e[119+NGN * 2] = 0.4450344476991877;
    C_e[120+NGN * 0] = 0.942775471590546;
    C_e[120+NGN * 1] = -0.9863088225120497;
    C_e[120+NGN * 2] = -0.023507956399415872;
    C_e[121+NGN * 0] = 1.1211987419834797;
    C_e[121+NGN * 1] = 0.001330628498995769;
    C_e[121+NGN * 2] = -0.006524034627153663;
    C_e[122+NGN * 0] = 19.923858345653784;
    C_e[122+NGN * 1] = 0.11397802425124724;
    C_e[122+NGN * 2] = 0.4525818462551493;
    C_e[123+NGN * 0] = 0.930621080474316;
    C_e[123+NGN * 1] = -0.986519092962776;
    C_e[123+NGN * 2] = -0.023184607340784688;
    C_e[124+NGN * 0] = 1.1202683789979642;
    C_e[124+NGN * 1] = 0.0013139059197446194;
    C_e[124+NGN * 2] = -0.006440779454311417;
    C_e[125+NGN * 0] = 19.66971797554723;
    C_e[125+NGN * 1] = 0.11241103274645366;
    C_e[125+NGN * 2] = 0.4601769908108186;
    C_e[126+NGN * 0] = 0.9183666606250948;
    C_e[126+NGN * 1] = -0.9867298030693656;
    C_e[126+NGN * 2] = -0.022859336057065364;
    C_e[127+NGN * 0] = 1.1193057171914196;
    C_e[127+NGN * 1] = 0.0012970166290237754;
    C_e[127+NGN * 2] = -0.006356743288080955;
    C_e[128+NGN * 0] = 19.413189237176866;
    C_e[128+NGN * 1] = 0.11083363298809572;
    C_e[128+NGN * 2] = 0.4678200757156566;
    C_e[129+NGN * 0] = 0.9060117246751008;
    C_e[129+NGN * 1] = -0.986940954461529;
    C_e[129+NGN * 2] = -0.02253213531673649;
    C_e[130+NGN * 0] = 1.1183105312276571;
    C_e[130+NGN * 1] = 0.0012799597304658289;
    C_e[130+NGN * 2] = -0.00627192204606027;
    C_e[131+NGN * 0] = 19.15425963506624;
    C_e[131+NGN * 1] = 0.1092457814424809;
    C_e[131+NGN * 2] = 0.4755112957272893;
    C_e[132+NGN * 0] = 0.8935557837902427;
    C_e[132+NGN * 1] = -0.9871525487719464;
    C_e[132+NGN * 2] = -0.022202997876386545;
    C_e[133+NGN * 0] = 1.1172825947787255;
    C_e[133+NGN * 1] = 0.0012627343246176782;
    C_e[133+NGN * 2] = -0.006186311632283316;
    C_e[134+NGN * 0] = 18.89291663217438;
    C_e[134+NGN * 1] = 0.107647434478127;
    C_e[134+NGN * 2] = 0.4832508460102488;
    C_e[135+NGN * 0] = 0.8809983476699006;
    C_e[135+NGN * 1] = -0.9873645876362536;
    C_e[135+NGN * 2] = -0.021871916480785857;
    C_e[136+NGN * 0] = 1.1162216805224172;
    C_e[136+NGN * 1] = 0.0012453395089371566;
    C_e[136+NGN * 2] = -0.006099907937208379;
    C_e[137+NGN * 0] = 18.62914764985989;
    C_e[137+NGN * 1] = 0.10603854836599544;
    C_e[137+NGN * 2] = 0.49103892213469674;
    C_e[138+NGN * 0] = 0.8683389245467351;
    C_e[138+NGN * 1] = -0.9875770726930267;
    C_e[138+NGN * 2] = -0.021538883862959376;
    C_e[139+NGN * 0] = 1.1151275601397752;
    C_e[139+NGN * 1] = 0.0012277743777897065;
    C_e[139+NGN * 2] = -0.006012706837706682;
    C_e[140+NGN * 0] = 18.36294006784566;
    C_e[140+NGN * 1] = 0.10441907927972864;
    C_e[140+NGN * 2] = 0.49887572007512826;
    C_e[141+NGN * 0] = 0.8555770211865308;
    C_e[141+NGN * 1] = -0.9877900055837696;
    C_e[141+NGN * 2] = -0.021203892744260263;
    C_e[142+NGN * 0] = 1.1140000043126066;
    C_e[142+NGN * 1] = 0.001210038022445095;
    C_e[142+NGN * 2] = -0.005924704197051196;
    C_e[143+NGN * 0] = 18.094281224184293;
    C_e[143+NGN * 1] = 0.10278898329589128;
    C_e[143+NGN * 2] = 0.5067614362090572;
    C_e[144+NGN * 0] = 0.8427121428880692;
    C_e[144+NGN * 1] = -0.9880033879528975;
    C_e[144+NGN * 2] = -0.02086693583444433;
    C_e[145+NGN * 0] = 1.1128387827209978;
    C_e[145+NGN * 1] = 0.001192129531074178;
    C_e[145+NGN * 2] = -0.0058358958649056755;
    C_e[146+NGN * 0] = 17.823158415224157;
    C_e[146+NGN * 1] = 0.10114821639421562;
    C_e[146+NGN * 2] = 0.5146962673156822;
    C_e[147+NGN * 0] = 0.8297437934830355;
    C_e[147+NGN * 1] = -0.9882172214477244;
    C_e[147+NGN * 2] = -0.02052800583174525;
    C_e[148+NGN * 0] = 1.1116436640408331;
    C_e[148+NGN * 1] = 0.0011740479887457052;
    C_e[148+NGN * 2] = -0.005746277677313921;
    C_e[149+NGN * 0] = 17.549558895576144;
    C_e[149+NGN * 1] = 0.099496734457851;
    C_e[149+NGN * 2] = 0.5226804105745334;
    C_e[150+NGN * 0] = 0.8166714753359573;
    C_e[150+NGN * 1] = -0.9884315077184468;
    C_e[150+NGN * 2] = -0.02018709542295069;
    C_e[151+NGN * 0] = 1.110414415941318;
    C_e[151+NGN * 1] = 0.001155792477423173;
    C_e[151+NGN * 2] = -0.005655845456689257;
    C_e[152+NGN * 0] = 17.273469878081105;
    C_e[152+NGN * 1] = 0.09783449327361723;
    C_e[152+NGN * 2] = 0.5307140635640998;
    C_e[153+NGN * 0] = 0.8034946893441745;
    C_e[153+NGN * 1] = -0.9886462484181286;
    C_e[153+NGN * 2] = -0.019844197283479203;
    C_e[154+NGN * 0] = 1.1091508050825052;
    C_e[154+NGN * 1] = 0.001137362075961721;
    C_e[154+NGN * 2] = -0.005564595011804241;
    C_e[155+NGN * 0] = 16.994878533777985;
    C_e[155+NGN * 1] = 0.09616144853226226;
    C_e[155+NGN * 2] = 0.5387974242604374;
    C_e[156+NGN * 0] = 0.7902129349378436;
    C_e[156+NGN * 1] = -0.9888614452026874;
    C_e[156+NGN * 2] = -0.019499304077458004;
    C_e[157+NGN * 0] = 1.107852597112824;
    C_e[157+NGN * 1] = 0.0011187558601050778;
    C_e[157+NGN * 2] = -0.005472522137780601;
    C_e[158+NGN * 0] = 16.713771991872633;
    C_e[158+NGN * 1] = 0.09447755582872376;
    C_e[158+NGN * 2] = 0.5469306910357578;
    C_e[159+NGN * 0] = 0.7768257100799743;
    C_e[159+NGN * 1] = -0.9890770997308772;
    C_e[159+NGN * 2] = -0.01915240845780159;
    C_e[160+NGN * 0] = 1.1065195566666162;
    C_e[160+NGN * 1] = 0.0010999729024825503;
    C_e[160+NGN * 2] = -0.0053796226160794;
    C_e[161+NGN * 0] = 16.43013733970734;
    C_e[161+NGN * 1] = 0.09278277066239515;
    C_e[161+NGN * 2] = 0.5551140626569961;
    C_e[162+NGN * 0] = 0.7633325112664989;
    C_e[162+NGN * 1] = -0.9892932136642748;
    C_e[162+NGN * 2] = -0.018803503066291208;
    C_e[163+NGN * 0] = 1.1051514473616717;
    C_e[163+NGN * 1] = 0.001081012272606062;
    C_e[163+NGN * 2] = -0.005285892214491437;
    C_e[164+NGN * 0] = 16.14396162273106;
    C_e[164+NGN * 1] = 0.09107704843739556;
    C_e[164+NGN * 2] = 0.5633477382843609;
    C_e[165+NGN * 0] = 0.7497328335263769;
    C_e[165+NGN * 1] = -0.9895097886672624;
    C_e[165+NGN * 2] = -0.018452580533655175;
    C_e[166+NGN * 0] = 1.1037480317967718;
    C_e[166+NGN * 1] = 0.001061873036867237;
    C_e[166+NGN * 2] = -0.005191326687127876;
    C_e[167+NGN * 0] = 15.85523184447036;
    C_e[167+NGN * 1] = 0.08936034446284405;
    C_e[167+NGN * 2] = 0.5716319174698622;
    C_e[168+NGN * 0] = 0.7360261704217308;
    C_e[168+NGN * 1] = -0.9897268264070126;
    C_e[168+NGN * 2] = -0.018099633479650077;
    C_e[169+NGN * 0] = 1.1023090715492356;
    C_e[169+NGN * 1] = 0.001042554258534534;
    C_e[169+NGN * 2] = -0.005095921774411115;
    C_e[170+NGN * 0] = 15.563934966501105;
    C_e[170+NGN * 1] = 0.0876326139531382;
    C_e[170+NGN * 2] = 0.5799668001558208;
    C_e[171+NGN * 0] = 0.7222120140480186;
    C_e[171+NGN * 1] = -0.989944328553472;
    C_e[171+NGN * 2] = -0.01774465451314283;
    C_e[172+NGN * 0] = 1.100834327172469;
    C_e[172+NGN * 1] = 0.0010230549977504277;
    C_e[172+NGN * 2] = -0.004999673203065893;
    C_e[173+NGN * 0] = 15.27005790842082;
    C_e[173+NGN * 1] = 0.08589381202823683;
    C_e[173+NGN * 2] = 0.5883525866733571;
    C_e[174+NGN * 0] = 0.7082898550342386;
    C_e[174+NGN * 1] = -0.9901622967793456;
    C_e[174+NGN * 2] = -0.0173876362321936;
    C_e[175+NGN * 0] = 1.0993235581935203;
    C_e[175+NGN * 1] = 0.0010033743115286369;
    C_e[175+NGN * 2] = -0.004902576686110637;
    C_e[176+NGN * 0] = 14.97358754782185;
    C_e[176+NGN * 1] = 0.08414389371394697;
    C_e[176+NGN * 2] = 0.5967894777408591;
    C_e[177+NGN * 0] = 0.6942591825431698;
    C_e[177+NGN * 1] = -0.990380732760079;
    C_e[177+NGN * 2] = -0.017028571224139616;
    C_e[178+NGN * 0] = 1.097776523110639;
    C_e[178+NGN * 1] = 0.0009835112537514052;
    C_e[178+NGN * 2] = -0.004804627922849047;
    C_e[179+NGN * 0] = 14.674510720265213;
    C_e[179+NGN * 1] = 0.0823828139422153;
    C_e[179+NGN * 2] = 0.6052776744624305;
    C_e[180+NGN * 0] = 0.6801194842716467;
    C_e[180+NGN * 1] = -0.990599638173843;
    C_e[180+NGN * 2] = -0.016667452065679857;
    C_e[181+NGN * 0] = 1.0961929793908376;
    C_e[181+NGN * 1] = 0.00096346487516683;
    C_e[181+NGN * 2] = -0.0047058225988619315;
    C_e[182+NGN * 0] = 14.372814219255204;
    C_e[182+NGN * 1] = 0.08061052755142381;
    C_e[182+NGN * 2] = 0.6138173783263186;
    C_e[183+NGN * 0] = 0.6658702464508693;
    C_e[183+NGN * 1] = -0.990819014701516;
    C_e[183+NGN * 2] = -0.016304271322960636;
    C_e[184+NGN * 0] = 1.0945726834674632;
    C_e[184+NGN * 1] = 0.00094323422338624;
    C_e[184+NGN * 2] = -0.004606156385999284;
    C_e[185+NGN * 0] = 14.068484796214776;
    C_e[185+NGN * 1] = 0.07882698928668982;
    C_e[185+NGN * 2] = 0.62240879120332;
    C_e[186+NGN * 0] = 0.6515109538467487;
    C_e[186+NGN * 1] = -0.991038864026668;
    C_e[186+NGN * 2] = -0.015939021551662054;
    C_e[187+NGN * 0] = 1.0929153907377662;
    C_e[187+NGN * 1] = 0.0009228183428816254;
    C_e[187+NGN * 2] = -0.00450562494237261;
    C_e[188+NGN * 0] = 13.761509160461648;
    C_e[188+NGN * 1] = 0.07703215380017045;
    C_e[188+NGN * 2] = 0.6310521153451679;
    C_e[189+NGN * 0] = 0.6370410897602874;
    C_e[189+NGN * 1] = -0.9912591878355412;
    C_e[189+NGN * 2] = -0.01557169529708538;
    C_e[190+NGN * 0] = 1.09122085556048;
    C_e[190+NGN * 1] = 0.0009022162749831176;
    C_e[190+NGN * 2] = -0.004404223912347504;
    C_e[191+NGN * 0] = 13.451873979185192;
    C_e[191+NGN * 1] = 0.07522597565137149;
    C_e[191+NGN * 2] = 0.6397475533828962;
    C_e[192+NGN * 0] = 0.6224601360279963;
    C_e[192+NGN * 1] = -0.991479987817036;
    C_e[192+NGN * 2] = -0.015202285094241309;
    C_e[193+NGN * 0] = 1.0894888312534026;
    C_e[193+NGN * 1] = 0.0008814270578765208;
    C_e[193+NGN * 2] = -0.004301948926536477;
    C_e[194+NGN * 0] = 13.1395658774241;
    C_e[194+NGN * 1] = 0.07340840930746068;
    C_e[194+NGN * 2] = 0.648495308325184;
    C_e[195+NGN * 0] = 0.607767573022348;
    C_e[195+NGN * 1] = -0.9917012656626896;
    C_e[195+NGN * 2] = -0.014830783467939123;
    C_e[196+NGN * 0] = 1.087719070090983;
    C_e[196+NGN * 1] = 0.0008604497266008959;
    C_e[196+NGN * 2] = -0.004198795601792043;
    C_e[197+NGN * 0] = 12.824571438044805;
    C_e[197+NGN * 1] = 0.07157940914358554;
    C_e[197+NGN * 2] = 0.6572955835566783;
    C_e[198+NGN * 0] = 0.5929628796522651;
    C_e[198+NGN * 1] = -0.9919230230666608;
    C_e[198+NGN * 2] = -0.014457182932876804;
    C_e[199+NGN * 0] = 1.0859113233019142;
    C_e[199+NGN * 1] = 0.0008392833130461954;
    C_e[199+NGN * 2] = -0.004094759541200058;
    C_e[200+NGN * 0] = 12.5068772017207;
    C_e[200+NGN * 1] = 0.06973892944319561;
    C_e[200+NGN * 2] = 0.6661485828362955;
    C_e[201+NGN * 0] = 0.5780455333636473;
    C_e[201+NGN * 1] = -0.9921452617257114;
    C_e[201+NGN * 2] = -0.014081475993732002;
    C_e[202+NGN * 0] = 1.0840653410667302;
    C_e[202+NGN * 1] = 0.0008179268459509528;
    C_e[202+NGN * 2] = -0.003989836334073315;
    C_e[203+NGN * 0] = 12.186469666912153;
    C_e[203+NGN * 1] = 0.06788692439836924;
    C_e[203+NGN * 2] = 0.6750545102955022;
    C_e[204+NGN * 0] = 0.5630150101399329;
    C_e[204+NGN * 1] = -0.9923679833391876;
    C_e[204+NGN * 2] = -0.01370365514525398;
    C_e[205+NGN * 0] = 1.0821808725154085;
    C_e[205+NGN * 1] = 0.0007963793509000232;
    C_e[205+NGN * 2] = -0.003884021555945409;
    C_e[206+NGN * 0] = 11.863335289847294;
    C_e[206+NGN * 1] = 0.06602334811014499;
    C_e[206+NGN * 2] = 0.6840135704365732;
    C_e[207+NGN * 0] = 0.5478707845027002;
    C_e[207+NGN * 1] = -0.9925911896090022;
    C_e[207+NGN * 2] = -0.013323712872356444;
    C_e[208+NGN * 0] = 1.0802576657249767;
    C_e[208+NGN * 1] = 0.000774639850322379;
    C_e[208+NGN * 2] = -0.003777310768564857;
    C_e[209+NGN * 0] = 11.537460484503637;
    C_e[209+NGN * 1] = 0.06414815458885754;
    C_e[209+NGN * 2] = 0.6930259681308286;
    C_e[210+NGN * 0] = 0.5326123295123036;
    C_e[210+NGN * 1] = -0.9928148822396148;
    C_e[210+NGN * 2] = -0.012941641650211331;
    C_e[211+NGN * 0] = 1.0782954677171277;
    C_e[211+NGN * 1] = 0.0007527073634889593;
    C_e[211+NGN * 2] = -0.003669699519889484;
    C_e[212+NGN * 0] = 11.208831622590486;
    C_e[212+NGN * 1] = 0.0622612977544783;
    C_e[212+NGN * 2] = 0.7020919086168497;
    C_e[213+NGN * 0] = 0.5172391167685497;
    C_e[213+NGN * 1] = -0.9930390629380152;
    C_e[213+NGN * 2] = -0.012557433944343529;
    C_e[214+NGN * 0] = 1.0762940244558354;
    C_e[214+NGN * 1] = 0.000730580906510574;
    C_e[214+NGN * 2] = -0.0035611833440810786;
    C_e[215+NGN * 0] = 10.877435033532164;
    C_e[215+NGN * 1] = 0.06036273143696049;
    C_e[215+NGN * 2] = 0.7112115974986714;
    C_e[216+NGN * 0] = 0.5017506164114104;
    C_e[216+NGN * 1] = -0.993263733413702;
    C_e[216+NGN * 2] = -0.012171082210726504;
    C_e[217+NGN * 0] = 1.0742530808449822;
    C_e[217+NGN * 1] = 0.0007082594923358621;
    C_e[217+NGN * 2] = -0.003451757761500318;
    C_e[218+NGN * 0] = 10.543257004452071;
    C_e[218+NGN * 1] = 0.05845240937658907;
    C_e[218+NGN * 2] = 0.7203852407439543;
    C_e[219+NGN * 0] = 0.4861462971217738;
    C_e[219+NGN * 1] = -0.9934888953786648;
    C_e[219+NGN * 2] = -0.011782578895878948;
    C_e[220+NGN * 0] = 1.072172380725986;
    C_e[220+NGN * 1] = 0.0006857421307493068;
    C_e[220+NGN * 2] = -0.0033414182787019613;
    C_e[221+NGN * 0] = 10.206283780157571;
    C_e[221+NGN * 1] = 0.05653028522433526;
    C_e[221+NGN * 2] = 0.7296130446821334;
    C_e[222+NGN * 0] = 0.470425626122235;
    C_e[222+NGN * 1] = -0.9937145505473652;
    C_e[222+NGN * 2] = -0.011391916436962286;
    C_e[223+NGN * 0] = 1.070051666875437;
    C_e[223+NGN * 1] = 0.0006630278283693055;
    C_e[223+NGN * 2] = -0.0032301603884303206;
    C_e[224+NGN * 0] = 9.866501563125697;
    C_e[224+NGN * 1] = 0.05459631254221577;
    C_e[224+NGN * 2] = 0.7388952160025453;
    C_e[225+NGN * 0] = 0.4545880691779244;
    C_e[225+NGN * 1] = -0.9939407006367156;
    C_e[225+NGN * 2] = -0.01099908726187922;
    C_e[226+NGN * 0] = 1.0678906810027389;
    C_e[226+NGN * 1] = 0.000640115588646297;
    C_e[226+NGN * 2] = -0.0031179795696150044;
    C_e[227+NGN * 0] = 9.523896513489722;
    C_e[227+NGN * 1] = 0.05265044480365688;
    C_e[227+NGN * 2] = 0.748231961752532;
    C_e[228+NGN * 0] = 0.43863309059737593;
    C_e[228+NGN * 1] = -0.994167347366061;
    C_e[228+NGN * 2] = -0.010604083789373176;
    C_e[229+NGN * 0] = 1.065689163747756;
    C_e[229+NGN * 1] = 0.0006170044118609448;
    C_e[229+NGN * 2] = -0.003004871287366941;
    C_e[230+NGN * 0] = 9.178454749026576;
    C_e[230+NGN * 1] = 0.050692635393863136;
    C_e[230+NGN * 2] = 0.7576234893355237;
    C_e[231+NGN * 0] = 0.42256015323343454;
    C_e[231+NGN * 1] = -0.994394492457158;
    C_e[231+NGN * 2] = -0.01020689842912876;
    C_e[232+NGN * 0] = 1.0634468546784674;
    C_e[232+NGN * 1] = 0.0005936932951223785;
    C_e[232+NGN * 2] = -0.002890830992974683;
    C_e[233+NGN * 0] = 8.830162345145096;
    C_e[233+NGN * 1] = 0.04872283761019098;
    C_e[233+NGN * 2] = 0.7670700065090973;
    C_e[234+NGN * 0] = 0.4063687184842029;
    C_e[234+NGN * 1] = -0.9946221376341552;
    C_e[234+NGN * 2] = -0.009807523581873156;
    C_e[235+NGN * 0] = 1.061163492288626;
    C_e[235+NGN * 1] = 0.0005701812323664932;
    C_e[235+NGN * 2] = -0.002775854123900992;
    C_e[236+NGN * 0] = 8.479005334875172;
    C_e[236+NGN * 1] = 0.04674100466252718;
    C_e[236+NGN * 2] = 0.7765717213830131;
    C_e[237+NGN * 0] = 0.3900582462940287;
    C_e[237+NGN * 1] = -0.994850284623572;
    C_e[237+NGN * 2] = -0.009405951639478514;
    C_e[238+NGN * 0] = 1.058838813995424;
    C_e[238+NGN * 1] = 0.0005464672143543061;
    C_e[238+NGN * 2] = -0.0026599361037797095;
    C_e[239+NGN * 0] = 8.124969708857734;
    C_e[239+NGN * 1] = 0.04474708967367208;
    C_e[239+NGN * 2] = 0.7861288424172284;
    C_e[240+NGN * 0] = 0.3736281951545317;
    C_e[240+NGN * 1] = -0.9950789351542793;
    C_e[240+NGN * 2] = -0.009002174985065323;
    C_e[241+NGN * 0] = 1.0564725561371668;
    C_e[241+NGN * 1] = 0.0005225502286703733;
    C_e[241+NGN * 2] = -0.002543072342412916;
    C_e[242+NGN * 0] = 7.768041415335636;
    C_e[242+NGN * 1] = 0.04274104567972778;
    C_e[242+NGN * 2] = 0.7957415784198874;
    C_e[243+NGN * 0] = 0.3570780221056716;
    C_e[243+NGN * 1] = -0.9953080909574772;
    C_e[243+NGN * 2] = -0.00859618599310676;
    C_e[244+NGN * 0] = 1.0540644539709492;
    C_e[244+NGN * 1] = 0.0004984292597212648;
    C_e[244+NGN * 2] = -0.0024252582357683766;
    C_e[245+NGN * 0] = 7.408206360145417;
    C_e[245+NGN * 1] = 0.04072282563049124;
    C_e[245+NGN * 2] = 0.8054101385452892;
    C_e[246+NGN * 0] = 0.3404071827368564;
    C_e[246+NGN * 1] = -0.9955377537666747;
    C_e[246+NGN * 2] = -0.008187977029534043;
    C_e[247+NGN * 0] = 1.0516142416703422;
    C_e[247+NGN * 1] = 0.0004741032887341004;
    C_e[247+NGN * 2] = -0.00230648916597728;
    C_e[248+NGN * 0] = 7.045450406709946;
    C_e[248+NGN * 1] = 0.03869238238985227;
    C_e[248+NGN * 2] = 0.8151347322918312;
    C_e[249+NGN * 0] = 0.32361513118809243;
    C_e[249+NGN * 1] = -0.9957679253176688;
    C_e[249+NGN * 2] = -0.0077775404518427745;
    C_e[250+NGN * 0] = 1.0491216523230853;
    C_e[250+NGN * 1] = 0.0004495712937551449;
    C_e[250+NGN * 2] = -0.00218676050133227;
    C_e[251+NGN * 0] = 6.6797593760319725;
    C_e[251+NGN * 1] = 0.0366496687361966;
    C_e[251+NGN * 2] = 0.8249155694999296;
    C_e[252+NGN * 0] = 0.30670132015117507;
    C_e[252+NGN * 1] = -0.995998607348522;
    C_e[252+NGN * 2] = -0.007364868609200292;
    C_e[253+NGN * 0] = 1.046586417928784;
    C_e[253+NGN * 1] = 0.0004248322496484645;
    C_e[253+NGN * 2] = -0.002066067596285777;
    C_e[254+NGN * 0] = 6.311119046688577;
    C_e[254+NGN * 1] = 0.03459463736281382;
    C_e[254+NGN * 2] = 0.8347528603499162;
    C_e[255+NGN * 0] = 0.2896652008709215;
    C_e[255+NGN * 1] = -0.9962298015995412;
    C_e[255+NGN * 2] = -0.006949953842554021;
    C_e[256+NGN * 0] = 1.0440082693966162;
    C_e[256+NGN * 1] = 0.0003998851280946446;
    C_e[256+NGN * 2] = -0.0019444057914486416;
    C_e[257+NGN * 0] = 5.939515154826543;
    C_e[257+NGN * 1] = 0.03252724087831047;
    C_e[257+NGN * 2] = 0.8446468153599118;
    C_e[258+NGN * 0] = 0.27250622314644557;
    C_e[258+NGN * 1] = -0.9964615098132564;
    C_e[258+NGN * 2] = -0.006532788484740849;
    C_e[259+NGN * 0] = 1.0413869365430442;
    C_e[259+NGN * 1] = 0.0003747288975895693;
    C_e[259+NGN * 2] = -0.0018217704135890449;
    C_e[260+NGN * 0] = 5.564933394158628;
    C_e[260+NGN * 1] = 0.030447431807028164;
    C_e[260+NGN * 2] = 0.854597645383675;
    C_e[261+NGN * 0] = 0.2552238353324744;
    C_e[261+NGN * 1] = -0.9966937337343972;
    C_e[261+NGN * 2] = -0.00611336486059751;
    C_e[262+NGN * 0] = 1.038722148089534;
    C_e[262+NGN * 1] = 0.00034936252344326294;
    C_e[262+NGN * 2] = -0.0016981567756317378;
    C_e[263+NGN * 0] = 5.187359415960783;
    C_e[263+NGN * 1] = 0.028355162589466743;
    C_e[263+NGN * 2] = 0.8646055616084277;
    C_e[264+NGN * 0] = 0.2378174843407081;
    C_e[264+NGN * 1] = -0.9969264751098712;
    C_e[264+NGN * 2] = -0.005691675287072003;
    C_e[265+NGN * 0] = 1.036013631660283;
    C_e[265+NGN * 1] = 0.0003237849677787947;
    C_e[265+NGN * 2] = -0.001573560176657578;
    C_e[266+NGN * 0] = 4.806778829070287;
    C_e[266+NGN * 1] = 0.026250385582712622;
    C_e[266+NGN * 2] = 0.8746707755526565;
    C_e[267+NGN * 0] = 0.22028661564122212;
    C_e[267+NGN * 1] = -0.997159735688742;
    C_e[267+NGN * 2] = -0.005267712073336026;
    C_e[268+NGN * 0] = 1.033261113779954;
    C_e[268+NGN * 1] = 0.0002979951895312464;
    C_e[268+NGN * 2] = -0.0014479759019033738;
    C_e[269+NGN * 0] = 4.423177199884829;
    C_e[269+NGN * 1] = 0.02413305306087228;
    C_e[269+NGN * 2] = 0.8847934990638892;
    C_e[270+NGN * 0] = 0.20263067326391265;
    C_e[270+NGN * 1] = -0.9973935172222054;
    C_e[270+NGN * 2] = -0.004841467520898464;
    C_e[271+NGN * 0] = 1.0304643198714158;
    C_e[271+NGN * 1] = 0.00027199214444674415;
    C_e[271+NGN * 2] = -0.0013213992227620382;
    C_e[272+NGN * 0] = 4.036540052362541;
    C_e[272+NGN * 1] = 0.022003117215510872;
    C_e[272+NGN * 2] = 0.8949739443164472;
    C_e[273+NGN * 0] = 0.18484909979998537;
    C_e[273+NGN * 1] = -0.9976278214635672;
    C_e[273+NGN * 2] = -0.004412933923719893;
    C_e[274+NGN * 0] = 1.0276229742534928;
    C_e[274+NGN * 1] = 0.00024577478508155416;
    C_e[274+NGN * 2] = -0.0011938253967830551;
    C_e[275+NGN * 0] = 3.6468528680229806;
    C_e[275+NGN * 1] = 0.01986053015609613;
    C_e[275+NGN * 2] = 0.905212323809174;
    C_e[276+NGN * 0] = 0.1669413364034883;
    C_e[276+NGN * 1] = -0.9978626501682188;
    C_e[276+NGN * 2] = -0.003982103568328155;
    C_e[277+NGN * 0] = 1.024736800138721;
    C_e[277+NGN * 1] = 0.00021934206080124344;
    C_e[277+NGN * 2] = -0.0010652496676732625;
    C_e[278+NGN * 0] = 3.254101085949069;
    C_e[278+NGN * 1] = 0.017705243910447453;
    C_e[278+NGN * 2] = 0.9155088503631376;
    C_e[279+NGN * 0] = 0.14890682279288844;
    C_e[279+NGN * 1] = -0.9980980050936156;
    C_e[279+NGN * 2] = -0.003548968733934961;
    C_e[280+NGN * 0] = 1.0218055196311135;
    C_e[280+NGN * 1] = 0.00019269291777990595;
    C_e[280+NGN * 2] = -0.0009356672652979472;
    C_e[281+NGN * 0] = 2.8582701027900144;
    C_e[281+NGN * 1] = 0.015537210425190276;
    C_e[281+NGN * 2] = 0.92586373711931;
    C_e[282+NGN * 0] = 0.1307449972526927;
    C_e[282+NGN * 1] = -0.9983338879992516;
    C_e[282+NGN * 2] = -0.003113521692553566;
    C_e[283+NGN * 0] = 1.0188288537239316;
    C_e[283+NGN * 1] = 0.0001658262989994542;
    C_e[283+NGN * 2] = -0.0008050734056822683;
    C_e[284+NGN * 0] = 2.459345272765198;
    C_e[284+NGN * 1] = 0.013356381566215754;
    C_e[284+NGN * 2] = 0.93627719753622;
    C_e[285+NGN * 0] = 0.11245529663511354;
    C_e[285+NGN * 1] = -0.998570300646636;
    C_e[285+NGN * 2] = -0.0026757547091174976;
    C_e[286+NGN * 0] = 1.015806522297465;
    C_e[286+NGN * 1] = 0.00013874114424897787;
    C_e[286+NGN * 2] = -0.0006734632910129937;
    C_e[287+NGN * 0] = 2.057311907669049;
    C_e[287+NGN * 1] = 0.011162709119145766;
    C_e[287+NGN * 2] = 0.9467494453875812;
    C_e[288+NGN * 0] = 0.09403715636177966;
    C_e[288+NGN * 1] = -0.99880724479927;
    C_e[288+NGN * 2] = -0.0022356600416003555;
    C_e[289+NGN * 0] = 1.0127382441168205;
    C_e[289+NGN * 1] = 0.0001114363901241686;
    C_e[289+NGN * 2] = -0.0005408321096405665;
    C_e[290+NGN * 0] = 1.652155276876912;
    C_e[290+NGN * 1] = 0.008956144789803274;
    C_e[290+NGN * 2] = 0.9572806947598969;
    C_e[291+NGN * 0] = 0.07549001042549172;
    C_e[291+NGN * 1] = -0.9990447222226208;
    C_e[291+NGN * 2] = -0.0017932299411366815;
    C_e[292+NGN * 0] = 1.009623736829717;
    C_e[292+NGN * 1] = 0.00008391097002681286;
    C_e[292+NGN * 2] = -0.0004071750360814967;
    C_e[293+NGN * 0] = 1.243860607351909;
    C_e[293+NGN * 1] = 0.006736640204688083;
    C_e[293+NGN * 2] = 0.9678711600500356;
    C_e[294+NGN * 0] = 0.056813291392023915;
    C_e[294+NGN * 1] = -0.9992827346840985;
    C_e[294+NGN * 2] = -0.001348456652143909;
    C_e[295+NGN * 0] = 1.006462716964292;
    C_e[295+NGN * 1] = 0.00005616381416435209;
    C_e[295+NGN * 2] = -0.0002724872310210819;
    C_e[296+NGN * 0] = 0.8324130836528122;
    C_e[296+NGN * 1] = 0.004504146911458023;
    C_e[296+NGN * 2] = 0.9785210559627856;
    C_e[297+NGN * 0] = 0.03800643040197105;
    C_e[297+NGN * 1] = -0.99952128395303;
    C_e[297+NGN * 2] = -0.0009013324124453936;
    C_e[298+NGN * 0] = 1.0032548999269133;
    C_e[298+NGN * 1] = 0.000028193849549511928;
    C_e[298+NGN * 2] = -0.00013676384131646076;
    C_e[299+NGN * 0] = 0.4177978479429274;
    C_e[299+NGN * 1] = 0.0022586163794155786;
    C_e[299+NGN * 2] = 0.9892305975083806;
    C_e[300+NGN * 0] = 0.019068857172642076;
    C_e[300+NGN * 1] = -0.9997603718006348;
    C_e[300+NGN * 2] = -0.0004518494533945366;
    C_e[301+NGN * 0] = 1;
    C_e[302+NGN * 1] = 1;
    C_e[303+NGN * 2] = 1;
    C_e[304+NGN * 0] = -1;
    C_e[305+NGN * 1] = -1;
    C_e[306+NGN * 2] = -1;
    lg_e[0] = -10000000000;
    ug_e[0] = 1.267;
    lg_e[1] = -10000000000;
    ug_e[1] = 47.40291;
    lg_e[2] = -10000000000;
    lg_e[3] = -10000000000;
    lg_e[4] = -10000000000;
    ug_e[4] = 1.267;
    lg_e[5] = -10000000000;
    ug_e[5] = 47.40291;
    lg_e[6] = -10000000000;
    lg_e[7] = -10000000000;
    ug_e[7] = 1.267;
    lg_e[8] = -10000000000;
    ug_e[8] = 47.40291;
    lg_e[9] = -10000000000;
    lg_e[10] = -10000000000;
    ug_e[10] = 1.267;
    lg_e[11] = -10000000000;
    ug_e[11] = 47.40291;
    lg_e[12] = -10000000000;
    lg_e[13] = -10000000000;
    ug_e[13] = 1.267;
    lg_e[14] = -10000000000;
    ug_e[14] = 47.40291;
    lg_e[15] = -10000000000;
    lg_e[16] = -10000000000;
    ug_e[16] = 1.267;
    lg_e[17] = -10000000000;
    ug_e[17] = 47.40291;
    lg_e[18] = -10000000000;
    lg_e[19] = -10000000000;
    ug_e[19] = 1.267;
    lg_e[20] = -10000000000;
    ug_e[20] = 47.40291;
    lg_e[21] = -10000000000;
    lg_e[22] = -10000000000;
    ug_e[22] = 1.267;
    lg_e[23] = -10000000000;
    ug_e[23] = 47.40291;
    lg_e[24] = -10000000000;
    lg_e[25] = -10000000000;
    ug_e[25] = 1.267;
    lg_e[26] = -10000000000;
    ug_e[26] = 47.40291;
    lg_e[27] = -10000000000;
    lg_e[28] = -10000000000;
    ug_e[28] = 1.267;
    lg_e[29] = -10000000000;
    ug_e[29] = 47.40291;
    lg_e[30] = -10000000000;
    lg_e[31] = -10000000000;
    ug_e[31] = 1.267;
    lg_e[32] = -10000000000;
    ug_e[32] = 47.40291;
    lg_e[33] = -10000000000;
    lg_e[34] = -10000000000;
    ug_e[34] = 1.267;
    lg_e[35] = -10000000000;
    ug_e[35] = 47.40291;
    lg_e[36] = -10000000000;
    lg_e[37] = -10000000000;
    ug_e[37] = 1.267;
    lg_e[38] = -10000000000;
    ug_e[38] = 47.40291;
    lg_e[39] = -10000000000;
    lg_e[40] = -10000000000;
    ug_e[40] = 1.267;
    lg_e[41] = -10000000000;
    ug_e[41] = 47.40291;
    lg_e[42] = -10000000000;
    lg_e[43] = -10000000000;
    ug_e[43] = 1.267;
    lg_e[44] = -10000000000;
    ug_e[44] = 47.40291;
    lg_e[45] = -10000000000;
    lg_e[46] = -10000000000;
    ug_e[46] = 1.267;
    lg_e[47] = -10000000000;
    ug_e[47] = 47.40291;
    lg_e[48] = -10000000000;
    lg_e[49] = -10000000000;
    ug_e[49] = 1.267;
    lg_e[50] = -10000000000;
    ug_e[50] = 47.40291;
    lg_e[51] = -10000000000;
    lg_e[52] = -10000000000;
    ug_e[52] = 1.267;
    lg_e[53] = -10000000000;
    ug_e[53] = 47.40291;
    lg_e[54] = -10000000000;
    lg_e[55] = -10000000000;
    ug_e[55] = 1.267;
    lg_e[56] = -10000000000;
    ug_e[56] = 47.40291;
    lg_e[57] = -10000000000;
    lg_e[58] = -10000000000;
    ug_e[58] = 1.267;
    lg_e[59] = -10000000000;
    ug_e[59] = 47.40291;
    lg_e[60] = -10000000000;
    lg_e[61] = -10000000000;
    ug_e[61] = 1.267;
    lg_e[62] = -10000000000;
    ug_e[62] = 47.40291;
    lg_e[63] = -10000000000;
    lg_e[64] = -10000000000;
    ug_e[64] = 1.267;
    lg_e[65] = -10000000000;
    ug_e[65] = 47.40291;
    lg_e[66] = -10000000000;
    lg_e[67] = -10000000000;
    ug_e[67] = 1.267;
    lg_e[68] = -10000000000;
    ug_e[68] = 47.40291;
    lg_e[69] = -10000000000;
    lg_e[70] = -10000000000;
    ug_e[70] = 1.267;
    lg_e[71] = -10000000000;
    ug_e[71] = 47.40291;
    lg_e[72] = -10000000000;
    lg_e[73] = -10000000000;
    ug_e[73] = 1.267;
    lg_e[74] = -10000000000;
    ug_e[74] = 47.40291;
    lg_e[75] = -10000000000;
    lg_e[76] = -10000000000;
    ug_e[76] = 1.267;
    lg_e[77] = -10000000000;
    ug_e[77] = 47.40291;
    lg_e[78] = -10000000000;
    lg_e[79] = -10000000000;
    ug_e[79] = 1.267;
    lg_e[80] = -10000000000;
    ug_e[80] = 47.40291;
    lg_e[81] = -10000000000;
    lg_e[82] = -10000000000;
    ug_e[82] = 1.267;
    lg_e[83] = -10000000000;
    ug_e[83] = 47.40291;
    lg_e[84] = -10000000000;
    lg_e[85] = -10000000000;
    ug_e[85] = 1.267;
    lg_e[86] = -10000000000;
    ug_e[86] = 47.40291;
    lg_e[87] = -10000000000;
    lg_e[88] = -10000000000;
    ug_e[88] = 1.267;
    lg_e[89] = -10000000000;
    ug_e[89] = 47.40291;
    lg_e[90] = -10000000000;
    lg_e[91] = -10000000000;
    ug_e[91] = 1.267;
    lg_e[92] = -10000000000;
    ug_e[92] = 47.40291;
    lg_e[93] = -10000000000;
    lg_e[94] = -10000000000;
    ug_e[94] = 1.267;
    lg_e[95] = -10000000000;
    ug_e[95] = 47.40291;
    lg_e[96] = -10000000000;
    lg_e[97] = -10000000000;
    ug_e[97] = 1.267;
    lg_e[98] = -10000000000;
    ug_e[98] = 47.40291;
    lg_e[99] = -10000000000;
    lg_e[100] = -10000000000;
    ug_e[100] = 1.267;
    lg_e[101] = -10000000000;
    ug_e[101] = 47.40291;
    lg_e[102] = -10000000000;
    lg_e[103] = -10000000000;
    ug_e[103] = 1.267;
    lg_e[104] = -10000000000;
    ug_e[104] = 47.40291;
    lg_e[105] = -10000000000;
    lg_e[106] = -10000000000;
    ug_e[106] = 1.267;
    lg_e[107] = -10000000000;
    ug_e[107] = 47.40291;
    lg_e[108] = -10000000000;
    lg_e[109] = -10000000000;
    ug_e[109] = 1.267;
    lg_e[110] = -10000000000;
    ug_e[110] = 47.40291;
    lg_e[111] = -10000000000;
    lg_e[112] = -10000000000;
    ug_e[112] = 1.267;
    lg_e[113] = -10000000000;
    ug_e[113] = 47.40291;
    lg_e[114] = -10000000000;
    lg_e[115] = -10000000000;
    ug_e[115] = 1.267;
    lg_e[116] = -10000000000;
    ug_e[116] = 47.40291;
    lg_e[117] = -10000000000;
    lg_e[118] = -10000000000;
    ug_e[118] = 1.267;
    lg_e[119] = -10000000000;
    ug_e[119] = 47.40291;
    lg_e[120] = -10000000000;
    lg_e[121] = -10000000000;
    ug_e[121] = 1.267;
    lg_e[122] = -10000000000;
    ug_e[122] = 47.40291;
    lg_e[123] = -10000000000;
    lg_e[124] = -10000000000;
    ug_e[124] = 1.267;
    lg_e[125] = -10000000000;
    ug_e[125] = 47.40291;
    lg_e[126] = -10000000000;
    lg_e[127] = -10000000000;
    ug_e[127] = 1.267;
    lg_e[128] = -10000000000;
    ug_e[128] = 47.40291;
    lg_e[129] = -10000000000;
    lg_e[130] = -10000000000;
    ug_e[130] = 1.267;
    lg_e[131] = -10000000000;
    ug_e[131] = 47.40291;
    lg_e[132] = -10000000000;
    lg_e[133] = -10000000000;
    ug_e[133] = 1.267;
    lg_e[134] = -10000000000;
    ug_e[134] = 47.40291;
    lg_e[135] = -10000000000;
    lg_e[136] = -10000000000;
    ug_e[136] = 1.267;
    lg_e[137] = -10000000000;
    ug_e[137] = 47.40291;
    lg_e[138] = -10000000000;
    lg_e[139] = -10000000000;
    ug_e[139] = 1.267;
    lg_e[140] = -10000000000;
    ug_e[140] = 47.40291;
    lg_e[141] = -10000000000;
    lg_e[142] = -10000000000;
    ug_e[142] = 1.267;
    lg_e[143] = -10000000000;
    ug_e[143] = 47.40291;
    lg_e[144] = -10000000000;
    lg_e[145] = -10000000000;
    ug_e[145] = 1.267;
    lg_e[146] = -10000000000;
    ug_e[146] = 47.40291;
    lg_e[147] = -10000000000;
    lg_e[148] = -10000000000;
    ug_e[148] = 1.267;
    lg_e[149] = -10000000000;
    ug_e[149] = 47.40291;
    lg_e[150] = -10000000000;
    lg_e[151] = -10000000000;
    ug_e[151] = 1.267;
    lg_e[152] = -10000000000;
    ug_e[152] = 47.40291;
    lg_e[153] = -10000000000;
    lg_e[154] = -10000000000;
    ug_e[154] = 1.267;
    lg_e[155] = -10000000000;
    ug_e[155] = 47.40291;
    lg_e[156] = -10000000000;
    lg_e[157] = -10000000000;
    ug_e[157] = 1.267;
    lg_e[158] = -10000000000;
    ug_e[158] = 47.40291;
    lg_e[159] = -10000000000;
    lg_e[160] = -10000000000;
    ug_e[160] = 1.267;
    lg_e[161] = -10000000000;
    ug_e[161] = 47.40291;
    lg_e[162] = -10000000000;
    lg_e[163] = -10000000000;
    ug_e[163] = 1.267;
    lg_e[164] = -10000000000;
    ug_e[164] = 47.40291;
    lg_e[165] = -10000000000;
    lg_e[166] = -10000000000;
    ug_e[166] = 1.267;
    lg_e[167] = -10000000000;
    ug_e[167] = 47.40291;
    lg_e[168] = -10000000000;
    lg_e[169] = -10000000000;
    ug_e[169] = 1.267;
    lg_e[170] = -10000000000;
    ug_e[170] = 47.40291;
    lg_e[171] = -10000000000;
    lg_e[172] = -10000000000;
    ug_e[172] = 1.267;
    lg_e[173] = -10000000000;
    ug_e[173] = 47.40291;
    lg_e[174] = -10000000000;
    lg_e[175] = -10000000000;
    ug_e[175] = 1.267;
    lg_e[176] = -10000000000;
    ug_e[176] = 47.40291;
    lg_e[177] = -10000000000;
    lg_e[178] = -10000000000;
    ug_e[178] = 1.267;
    lg_e[179] = -10000000000;
    ug_e[179] = 47.40291;
    lg_e[180] = -10000000000;
    lg_e[181] = -10000000000;
    ug_e[181] = 1.267;
    lg_e[182] = -10000000000;
    ug_e[182] = 47.40291;
    lg_e[183] = -10000000000;
    lg_e[184] = -10000000000;
    ug_e[184] = 1.267;
    lg_e[185] = -10000000000;
    ug_e[185] = 47.40291;
    lg_e[186] = -10000000000;
    lg_e[187] = -10000000000;
    ug_e[187] = 1.267;
    lg_e[188] = -10000000000;
    ug_e[188] = 47.40291;
    lg_e[189] = -10000000000;
    lg_e[190] = -10000000000;
    ug_e[190] = 1.267;
    lg_e[191] = -10000000000;
    ug_e[191] = 47.40291;
    lg_e[192] = -10000000000;
    lg_e[193] = -10000000000;
    ug_e[193] = 1.267;
    lg_e[194] = -10000000000;
    ug_e[194] = 47.40291;
    lg_e[195] = -10000000000;
    lg_e[196] = -10000000000;
    ug_e[196] = 1.267;
    lg_e[197] = -10000000000;
    ug_e[197] = 47.40291;
    lg_e[198] = -10000000000;
    lg_e[199] = -10000000000;
    ug_e[199] = 1.267;
    lg_e[200] = -10000000000;
    ug_e[200] = 47.40291;
    lg_e[201] = -10000000000;
    lg_e[202] = -10000000000;
    ug_e[202] = 1.267;
    lg_e[203] = -10000000000;
    ug_e[203] = 47.40291;
    lg_e[204] = -10000000000;
    lg_e[205] = -10000000000;
    ug_e[205] = 1.267;
    lg_e[206] = -10000000000;
    ug_e[206] = 47.40291;
    lg_e[207] = -10000000000;
    lg_e[208] = -10000000000;
    ug_e[208] = 1.267;
    lg_e[209] = -10000000000;
    ug_e[209] = 47.40291;
    lg_e[210] = -10000000000;
    lg_e[211] = -10000000000;
    ug_e[211] = 1.267;
    lg_e[212] = -10000000000;
    ug_e[212] = 47.40291;
    lg_e[213] = -10000000000;
    lg_e[214] = -10000000000;
    ug_e[214] = 1.267;
    lg_e[215] = -10000000000;
    ug_e[215] = 47.40291;
    lg_e[216] = -10000000000;
    lg_e[217] = -10000000000;
    ug_e[217] = 1.267;
    lg_e[218] = -10000000000;
    ug_e[218] = 47.40291;
    lg_e[219] = -10000000000;
    lg_e[220] = -10000000000;
    ug_e[220] = 1.267;
    lg_e[221] = -10000000000;
    ug_e[221] = 47.40291;
    lg_e[222] = -10000000000;
    lg_e[223] = -10000000000;
    ug_e[223] = 1.267;
    lg_e[224] = -10000000000;
    ug_e[224] = 47.40291;
    lg_e[225] = -10000000000;
    lg_e[226] = -10000000000;
    ug_e[226] = 1.267;
    lg_e[227] = -10000000000;
    ug_e[227] = 47.40291;
    lg_e[228] = -10000000000;
    lg_e[229] = -10000000000;
    ug_e[229] = 1.267;
    lg_e[230] = -10000000000;
    ug_e[230] = 47.40291;
    lg_e[231] = -10000000000;
    lg_e[232] = -10000000000;
    ug_e[232] = 1.267;
    lg_e[233] = -10000000000;
    ug_e[233] = 47.40291;
    lg_e[234] = -10000000000;
    lg_e[235] = -10000000000;
    ug_e[235] = 1.267;
    lg_e[236] = -10000000000;
    ug_e[236] = 47.40291;
    lg_e[237] = -10000000000;
    lg_e[238] = -10000000000;
    ug_e[238] = 1.267;
    lg_e[239] = -10000000000;
    ug_e[239] = 47.40291;
    lg_e[240] = -10000000000;
    lg_e[241] = -10000000000;
    ug_e[241] = 1.267;
    lg_e[242] = -10000000000;
    ug_e[242] = 47.40291;
    lg_e[243] = -10000000000;
    lg_e[244] = -10000000000;
    ug_e[244] = 1.267;
    lg_e[245] = -10000000000;
    ug_e[245] = 47.40291;
    lg_e[246] = -10000000000;
    lg_e[247] = -10000000000;
    ug_e[247] = 1.267;
    lg_e[248] = -10000000000;
    ug_e[248] = 47.40291;
    lg_e[249] = -10000000000;
    lg_e[250] = -10000000000;
    ug_e[250] = 1.267;
    lg_e[251] = -10000000000;
    ug_e[251] = 47.40291;
    lg_e[252] = -10000000000;
    lg_e[253] = -10000000000;
    ug_e[253] = 1.267;
    lg_e[254] = -10000000000;
    ug_e[254] = 47.40291;
    lg_e[255] = -10000000000;
    lg_e[256] = -10000000000;
    ug_e[256] = 1.267;
    lg_e[257] = -10000000000;
    ug_e[257] = 47.40291;
    lg_e[258] = -10000000000;
    lg_e[259] = -10000000000;
    ug_e[259] = 1.267;
    lg_e[260] = -10000000000;
    ug_e[260] = 47.40291;
    lg_e[261] = -10000000000;
    lg_e[262] = -10000000000;
    ug_e[262] = 1.267;
    lg_e[263] = -10000000000;
    ug_e[263] = 47.40291;
    lg_e[264] = -10000000000;
    lg_e[265] = -10000000000;
    ug_e[265] = 1.267;
    lg_e[266] = -10000000000;
    ug_e[266] = 47.40291;
    lg_e[267] = -10000000000;
    lg_e[268] = -10000000000;
    ug_e[268] = 1.267;
    lg_e[269] = -10000000000;
    ug_e[269] = 47.40291;
    lg_e[270] = -10000000000;
    lg_e[271] = -10000000000;
    ug_e[271] = 1.267;
    lg_e[272] = -10000000000;
    ug_e[272] = 47.40291;
    lg_e[273] = -10000000000;
    lg_e[274] = -10000000000;
    ug_e[274] = 1.267;
    lg_e[275] = -10000000000;
    ug_e[275] = 47.40291;
    lg_e[276] = -10000000000;
    lg_e[277] = -10000000000;
    ug_e[277] = 1.267;
    lg_e[278] = -10000000000;
    ug_e[278] = 47.40291;
    lg_e[279] = -10000000000;
    lg_e[280] = -10000000000;
    ug_e[280] = 1.267;
    lg_e[281] = -10000000000;
    ug_e[281] = 47.40291;
    lg_e[282] = -10000000000;
    lg_e[283] = -10000000000;
    ug_e[283] = 1.267;
    lg_e[284] = -10000000000;
    ug_e[284] = 47.40291;
    lg_e[285] = -10000000000;
    lg_e[286] = -10000000000;
    ug_e[286] = 1.267;
    lg_e[287] = -10000000000;
    ug_e[287] = 47.40291;
    lg_e[288] = -10000000000;
    lg_e[289] = -10000000000;
    ug_e[289] = 1.267;
    lg_e[290] = -10000000000;
    ug_e[290] = 47.40291;
    lg_e[291] = -10000000000;
    lg_e[292] = -10000000000;
    ug_e[292] = 1.267;
    lg_e[293] = -10000000000;
    ug_e[293] = 47.40291;
    lg_e[294] = -10000000000;
    lg_e[295] = -10000000000;
    ug_e[295] = 1.267;
    lg_e[296] = -10000000000;
    ug_e[296] = 47.40291;
    lg_e[297] = -10000000000;
    lg_e[298] = -10000000000;
    ug_e[298] = 1.267;
    lg_e[299] = -10000000000;
    ug_e[299] = 47.40291;
    lg_e[300] = -10000000000;
    lg_e[301] = -10000000000;
    ug_e[301] = 1.267;
    lg_e[302] = -10000000000;
    ug_e[302] = 90;
    lg_e[303] = -10000000000;
    ug_e[303] = 47.40291;
    lg_e[304] = -10000000000;
    lg_e[305] = -10000000000;
    lg_e[306] = -10000000000;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "C", C_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lg", lg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ug", ug_e);
    free(C_e);
    free(lug_e);


}


static void turbine_acados_create_set_opts(turbine_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    double newton_tol_val = 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_tol", &newton_tol_val);

    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 600;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    int log_primal_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_primal_step_norm", &log_primal_step_norm);

    double nlp_solver_tol_min_step_norm = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_min_step_norm", &nlp_solver_tol_min_step_norm);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    // set SQP specific options
    double nlp_solver_tol_stat = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 100;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    // set options for adaptive Levenberg-Marquardt Update
    bool with_adaptive_levenberg_marquardt = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_adaptive_levenberg_marquardt", &with_adaptive_levenberg_marquardt);

    double adaptive_levenberg_marquardt_lam = 5;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_lam", &adaptive_levenberg_marquardt_lam);

    double adaptive_levenberg_marquardt_mu_min = 0.0000000000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu_min", &adaptive_levenberg_marquardt_mu_min);

    double adaptive_levenberg_marquardt_mu0 = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu0", &adaptive_levenberg_marquardt_mu0);

    bool eval_residual_at_max_iter = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "eval_residual_at_max_iter", &eval_residual_at_max_iter);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for turbine_acados_create: step 7
 */
void turbine_acados_set_nlp_out(turbine_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    x0[0] = 0.000001;
    x0[1] = 0.000001;
    x0[2] = 0.000001;


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for turbine_acados_create: step 9
 */
int turbine_acados_create_precompute(turbine_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int turbine_acados_create_with_discretization(turbine_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != TURBINE_N && !new_time_steps) {
        fprintf(stderr, "turbine_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, TURBINE_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    turbine_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = turbine_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    turbine_acados_create_set_opts(capsule);

    // 4) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 5) setup functions, nlp_in and default parameters
    turbine_acados_create_setup_functions(capsule);
    turbine_acados_setup_nlp_in(capsule, N, new_time_steps);
    turbine_acados_create_set_default_parameters(capsule);

    // 6) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // 7) create and set nlp_out
    // 7.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 7.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    turbine_acados_set_nlp_out(capsule);

    // 8) do precomputations
    int status = turbine_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int turbine_acados_update_qp_solver_cond_N(turbine_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from turbine_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = turbine_acados_create_precompute(capsule);
    return status;
}


int turbine_acados_reset(turbine_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "pi", buffer);
            ocp_nlp_set(nlp_solver, i, "xdot_guess", buffer);
            ocp_nlp_set(nlp_solver, i, "z_guess", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int turbine_acados_update_params(turbine_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 0;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int turbine_acados_update_params_sparse(turbine_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int turbine_acados_set_p_global_and_precompute_dependencies(turbine_solver_capsule* capsule, double* data, int data_len)
{

    printf("No global_data, turbine_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int turbine_acados_solve(turbine_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int turbine_acados_setup_qp_matrices_and_factorize(turbine_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



void turbine_acados_batch_solve(turbine_solver_capsule ** capsules, int * status_out, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_solve(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}


void turbine_acados_batch_setup_qp_matrices_and_factorize(turbine_solver_capsule ** capsules, int * status_out, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_setup_qp_matrices_and_factorize(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}


void turbine_acados_batch_eval_params_jac(turbine_solver_capsule ** capsules, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_params_jac(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}



void turbine_acados_batch_eval_solution_sens_adj_p(turbine_solver_capsule ** capsules, const char *field, int stage, double *out, int offset, int N_batch)
{


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_solution_sens_adj_p(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->sens_out, field, stage, out + i*offset);
    }


    return;
}


void turbine_acados_batch_set_flat(turbine_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_set_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_set_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }


    return;
}



void turbine_acados_batch_get_flat(turbine_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_get_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_get_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }


    return;
}


int turbine_acados_free(turbine_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->impl_dae_fun[i]);
        external_function_external_param_casadi_free(&capsule->impl_dae_fun_jac_x_xdot_z[i]);
        external_function_external_param_casadi_free(&capsule->impl_dae_jac_x_xdot_u_z[i]);
    }
    free(capsule->impl_dae_fun);
    free(capsule->impl_dae_fun_jac_x_xdot_z);
    free(capsule->impl_dae_jac_x_xdot_u_z);

    // cost

    // constraints



    return 0;
}


void turbine_acados_print_stats(turbine_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    double stat[1200];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j == 5 || j == 6)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
}

int turbine_acados_custom_update(turbine_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *turbine_acados_get_nlp_in(turbine_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *turbine_acados_get_nlp_out(turbine_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *turbine_acados_get_sens_out(turbine_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *turbine_acados_get_nlp_solver(turbine_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *turbine_acados_get_nlp_config(turbine_solver_capsule* capsule) { return capsule->nlp_config; }
void *turbine_acados_get_nlp_opts(turbine_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *turbine_acados_get_nlp_dims(turbine_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *turbine_acados_get_nlp_plan(turbine_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
