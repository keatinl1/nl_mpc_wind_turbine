// std
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// blasfeo
#include "blasfeo_d_aux_ext_dep.h"
#include "blasfeo_i_aux_ext_dep.h"

// acados
#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados/utils/mem.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"

// example specific
#include "examples/c/wt_model_nx6/nx6p2/wt_model.h"
#include "examples/c/wt_model_nx6/setup.c"

#define NN 10 // horizon length

#define MAX_SQP_ITERS 10
#define NREP 1

static void shift_states(ocp_nlp_dims *dims, ocp_nlp_out *out, double *x_end)
{
    int N = dims->N;

    for (int i = 0; i < N; i++)
         blasfeo_dveccp(dims->nx[i], &out->ux[i], dims->nu[i], &out->ux[i+1], dims->nu[i+1]);
     blasfeo_pack_dvec(dims->nx[N], x_end, 1, &out->ux[N], dims->nu[N]);
}

static void shift_controls(ocp_nlp_dims *dims, ocp_nlp_out *out, double *u_end)
{
    int N = dims->N;

    for (int i = 0; i < N-1; i++)
         blasfeo_dveccp(dims->nu[i], &out->ux[i], 0, &out->ux[i+1], 0);
     blasfeo_pack_dvec(dims->nu[N-1], u_end, 1, &out->ux[N-1], 0);
}

static void select_dynamics_wt_casadi(int N,external_function_param_casadi *expl_vde_for)
{
    for (int ii = 0; ii < N; ii++)
    {
        expl_vde_for[ii].casadi_fun = &wt_nx6p2_expl_vde_for;
        expl_vde_for[ii].casadi_work = &wt_nx6p2_expl_vde_for_work;
        expl_vde_for[ii].casadi_sparsity_in = &wt_nx6p2_expl_vde_for_sparsity_in;
        expl_vde_for[ii].casadi_sparsity_out = &wt_nx6p2_expl_vde_for_sparsity_out;
        expl_vde_for[ii].casadi_n_in = &wt_nx6p2_expl_vde_for_n_in;
        expl_vde_for[ii].casadi_n_out = &wt_nx6p2_expl_vde_for_n_out;
    }
}

void save_states_to_csv(double *x_sim, double *u_sim, int num_timesteps, int nx_, int nu_) 
{
    FILE *file = fopen("simulation_results.csv", "w");

    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    // Write header
    fprintf(file, "TimeStep");
    for (int j = 0; j < nx_; j++) {
        fprintf(file, ",State_%d", j);
    }
    for (int j = 0; j < nu_; j++) {
        fprintf(file, ",Control_%d", j);
    }
    fprintf(file, "\n");

    // Write data
    for (int i = 0; i < num_timesteps; i++) {
        fprintf(file, "%d", i);
        for (int j = 0; j < nx_; j++) {
            fprintf(file, ",%lf", x_sim[i * nx_ + j]);
        }
        for (int j = 0; j < nu_; j++) {
            fprintf(file, ",%lf", u_sim[i * nu_ + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Simulation results saved to simulation_results.csv\n");
}


/************************************************
* main
************************************************/

int main()
{
    // _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
    int nx_ = 8;
    int nu_ = 2;
    int ny_ = 4;

    int np = 1; // number of local parametrs for each dynamics model function

    /************************************************
    * problem dimensions
    ************************************************/

    // optimization variables
    int nx[NN+1] = {}; // states
    int nu[NN+1] = {}; // inputs
    int nz[NN+1] = {}; // algebraic variables
    int ns[NN+1] = {}; // slacks
    // cost
    int ny[NN+1] = {}; // measurements
    // constraints
    int nbx[NN+1] = {}; // state bounds
    int nbu[NN+1] = {}; // input bounds
    int ng[NN+1] = {}; // general linear constraints

    // TODO(dimitris): setup bounds on states and controls based on ACADO controller
    nx[0] = nx_;
    nu[0] = nu_;
    nbx[0] = nx_;
    nbu[0] = nu_;
    ng[0] = 0;


    for (int i = 1; i < NN; i++)
    {
        nx[i] = nx_;
        nu[i] = nu_;
        nbx[i] = 3;
        nbu[i] = nu_;
        ng[i] = 0;
        ny[i] = 4;
        nz[i] = 0;
    }

    nx[NN] = nx_;
    nu[NN] = 0;
    nbx[NN] = 3;
    nbu[NN] = 0;
    ng[NN] = 0;
    ny[NN] = 2;
    nz[NN] = 0;

    /************************************************
    * problem data
    ************************************************/

    double *x_end = malloc(sizeof(double)*nx_);
    double *u_end = malloc(sizeof(double)*nu_);

    // value of last stage when shifting states and controls
    for (int i = 0; i < nx_; i++) x_end[i] = 0.0;
    for (int i = 0; i < nu_; i++) u_end[i] = 0.0;



    /* constraints */

    // pitch angle rate
    double dbeta_min = - 8.0;
    double dbeta_max =   8.0;
    // generator torque
    double dM_gen_min = - 1.0;
    double dM_gen_max =   1.0;
    // generator angular velocity
    double OmegaR_min =  6.0/60*2*3.14159265359;
    double OmegaR_max = 13.0/60*2*3.14159265359;
    // pitch angle
    double beta_min =  0.0;
    double beta_max = 35.0;
    // generator torque
    double M_gen_min = 0.0;
    double M_gen_max = 5.0;
    // electric power
    double Pel_min = 0.0;
    double Pel_max = 5.0;


    /* box constraints */

    // acados inf
    double acados_inf = 1e8;

    // first stage

    // input bounds
    int *idxbu0 = malloc(nbu[0]*sizeof(int));
    double *lbu0 = malloc((nbu[0])*sizeof(double));
    double *ubu0 = malloc((nbu[0])*sizeof(double));

    // pitch angle rate
    idxbu0[0] = 0;
    lbu0[0] = dbeta_min;
    ubu0[0] = dbeta_max;

    // generator torque
    idxbu0[1] = 1;
    lbu0[1] = dM_gen_min;
    ubu0[1] = dM_gen_max;

    // state bounds
    int *idxbx0 = malloc(nbx[0]*sizeof(int));
    double *lbx0 = malloc((nbx[0])*sizeof(double));
    double *ubx0 = malloc((nbx[0])*sizeof(double));

    // dummy
    for (int ii=0; ii<nbx[0]; ii++)
    {
        idxbx0[ii] = ii;
        lbx0[ii] = - acados_inf;
        ubx0[ii] =   acados_inf;
    }


    // middle stages

    // input bounds
    int *idxbu1 = malloc(nbu[1]*sizeof(int));
    double *lbu1 = malloc((nbu[1])*sizeof(double));
    double *ubu1 = malloc((nbu[1])*sizeof(double));

    // pitch angle rate
    idxbu1[0] = 0;
    lbu1[0] = dbeta_min;
    ubu1[0] = dbeta_max;

    // generator torque rate
    idxbu1[1] = 1;
    lbu1[1] = dM_gen_min;
    ubu1[1] = dM_gen_max;

    // state bounds
    int *idxbx1 = malloc(nbx[1]*sizeof(int));
    double *lbx1 = malloc((nbx[1])*sizeof(double));
    double *ubx1 = malloc((nbx[1])*sizeof(double));

    // generator angular velocity
    idxbx1[0] = 0;
    lbx1[0] = OmegaR_min;
    ubx1[0] = OmegaR_max;

    // pitch angle
    idxbx1[1] = 6;
    lbx1[1] = beta_min;
    ubx1[1] = beta_max;

    // generator torque
    idxbx1[2] = 7;
    lbx1[2] = M_gen_min;
    ubx1[2] = M_gen_max;

    // last stage

    // state bounds
    int *idxbxN = malloc(nbx[NN]*sizeof(int));
    double *lbxN = malloc((nbx[NN])*sizeof(double));
    double *ubxN = malloc((nbx[NN])*sizeof(double));

    // generator angular velocity
    idxbxN[0] = 0;
    lbxN[0] = OmegaR_min;
    ubxN[0] = OmegaR_max;

    // pitch angle
    idxbxN[1] = 6;
    lbxN[1] = beta_min;
    ubxN[1] = beta_max;

    // generator torque
    idxbxN[2] = 7;
    lbxN[2] = M_gen_min;
    ubxN[2] = M_gen_max;

    // to shift
    double *specific_u = malloc(nu_*sizeof(double));
    double *specific_x = malloc(nx_*sizeof(double));

    /* linear least squares */

    // output definition
    // y = {x[0], x[4]; u[0]; u[1]; u[2]};
    //   = Vx * x + Vu * u

    double *Vx = malloc((ny_*nx_)*sizeof(double));
    for (int ii=0; ii<ny_*nx_; ii++)
        Vx[ii] = 0.0;
    Vx[0+ny_*0] = 1.0;
    Vx[1+ny_*4] = 1.0;

    double *Vu = malloc((ny_*nu_)*sizeof(double));
    for (int ii=0; ii<ny_*nu_; ii++)
        Vu[ii] = 0.0;
    Vu[2+ny_*0] = 1.0;
    Vu[3+ny_*1] = 1.0;

    double *VxN = malloc((ny[NN]*nx[NN])*sizeof(double));
    for (int ii=0; ii<ny[NN]*nx[NN]; ii++)
        VxN[ii] = 0.0;
    VxN[0+ny[NN]*0] = 1.0;
    VxN[1+ny[NN]*4] = 1.0;


    double *W = malloc((ny_*ny_)*sizeof(double));
    for (int ii=0; ii<ny_*ny_; ii++)
        W[ii] = 0.0;
    W[0+ny_*0] = 1.5114;
    W[1+ny_*0] = -0.0649;
    W[0+ny_*1] = -0.0649;
    W[1+ny_*1] = 0.0180;
    W[2+ny_*2] = 0.01;
    W[3+ny_*3] = 0.001;

    double *W_N = malloc((ny[NN]*ny[NN])*sizeof(double));
    W_N[0+ny[NN]*0] = 1.5114;
    W_N[1+ny[NN]*0] = -0.0649;
    W_N[0+ny[NN]*1] = -0.0649;
    W_N[1+ny[NN]*1] = 0.0180;

    /* slacks */

    // first stage
    double *lZ0 = malloc(ns[0]*sizeof(double));
    double *uZ0 = malloc(ns[0]*sizeof(double));
    double *lz0 = malloc(ns[0]*sizeof(double));
    double *uz0 = malloc(ns[0]*sizeof(double));

    // middle stages
    double *lZ1 = malloc(ns[1]*sizeof(double));
    double *uZ1 = malloc(ns[1]*sizeof(double));
    double *lz1 = malloc(ns[1]*sizeof(double));
    double *uz1 = malloc(ns[1]*sizeof(double));
    lZ1[0] = 1e2;
    uZ1[0] = 1e2;
    lz1[0] = 0e1;
    uz1[0] = 0e1;

    // final stage
    double *lZN = malloc(ns[NN]*sizeof(double));
    double *uZN = malloc(ns[NN]*sizeof(double));
    double *lzN = malloc(ns[NN]*sizeof(double));
    double *uzN = malloc(ns[NN]*sizeof(double));

    /************************************************
    * plan + config
    ************************************************/

    ocp_nlp_plan_t *plan = ocp_nlp_plan_create(NN);

    plan->nlp_solver = SQP;

    for (int i = 0; i <= NN; i++)
        plan->nlp_cost[i] = LINEAR_LS;

    plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    for (int i = 0; i < NN; i++)
    {
        plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        plan->sim_solver_plan[i].sim_solver = ERK;
    }

    for (int i = 0; i <= NN; i++)
        plan->nlp_constraints[i] = BGH;

    ocp_nlp_config *config = ocp_nlp_config_create(*plan);

    /************************************************
    * ocp_nlp_dims
    ************************************************/

    ocp_nlp_dims *dims = ocp_nlp_dims_create(config);

    ocp_nlp_dims_set_opt_vars(config, dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(config, dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(config, dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(config, dims, "ns", ns);

    for (int i = 0; i <= NN; i++)
    {
        ocp_nlp_dims_set_cost(config, dims, i, "ny", &ny[i]);

        ocp_nlp_dims_set_constraints(config, dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(config, dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(config, dims, i, "ng", &ng[i]);
    }

    /************************************************
    * dynamics
    ************************************************/
    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);
    ext_fun_opts.external_workspace = true;

    // explicit model
    external_function_param_casadi *expl_vde_for = malloc(NN*sizeof(external_function_param_casadi));
    select_dynamics_wt_casadi(NN, expl_vde_for);

    // explicit model
    external_function_param_casadi_create_array(NN, expl_vde_for, np, &ext_fun_opts);

    /************************************************
    * nlp_in
    ************************************************/

    ocp_nlp_in *nlp_in = ocp_nlp_in_create(config, dims);

    // sampling times
    for (int ii=0; ii<NN; ii++)
    {
        nlp_in->Ts[ii] = 0.2;
    }

    // output definition: y = [x; u]

    /* cost */

    // linear ls
    int status = ACADOS_SUCCESS;

    for (int i = 0; i <= NN; i++)
    {
        // Cyt
        ocp_nlp_cost_model_set(config, dims, nlp_in, i, "Vu", Vu);
        if (i < NN)
            ocp_nlp_cost_model_set(config, dims, nlp_in, i, "Vx", Vx);
        else
            ocp_nlp_cost_model_set(config, dims, nlp_in, i, "Vx", VxN);

        // W
        ocp_nlp_cost_model_set(config, dims, nlp_in, i, "W", W);
    }
    status = ocp_nlp_cost_model_set(config, dims, nlp_in, NN, "W", W_N);

    // slacks (middle stages)
    for (int ii=1; ii<NN; ii++)
    {
        ocp_nlp_cost_model_set(config, dims, nlp_in, ii, "Zl", lZ1);
        ocp_nlp_cost_model_set(config, dims, nlp_in, ii, "Zu", uZ1);
        ocp_nlp_cost_model_set(config, dims, nlp_in, ii, "zl", lz1);
        ocp_nlp_cost_model_set(config, dims, nlp_in, ii, "zu", uz1);
    }


    /* dynamics */

    int set_fun_status;

    for (int i=0; i<NN; i++)
    {
        if (plan->sim_solver_plan[i].sim_solver == ERK)
        {
            set_fun_status = ocp_nlp_dynamics_model_set(config, dims, nlp_in, i, "expl_vde_for", &expl_vde_for[i]);
            if (set_fun_status != 0) exit(1);
        }
        else
        {
            printf("\nWrong sim name\n\n");
            exit(1);
        }
    }


    /* constraints */

    /* box constraints */

    // fist stage
    ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "idxbu", idxbu0);
    ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "lbu", lbu0);
    ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "ubu", ubu0);
    ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "ubx", ubx0);
    // middle stages
    for (int i = 1; i < NN; i++)
    {
        ocp_nlp_constraints_model_set(config, dims, nlp_in, i, "idxbu", idxbu1);
        ocp_nlp_constraints_model_set(config, dims, nlp_in, i, "lbu", lbu1);
        ocp_nlp_constraints_model_set(config, dims, nlp_in, i, "ubu", ubu1);
        ocp_nlp_constraints_model_set(config, dims, nlp_in, i, "idxbx", idxbx1);
        ocp_nlp_constraints_model_set(config, dims, nlp_in, i, "lbx", lbx1);
        ocp_nlp_constraints_model_set(config, dims, nlp_in, i, "ubx", ubx1);
    }
    // last stage
    ocp_nlp_constraints_model_set(config, dims, nlp_in, NN, "idxbx", idxbxN);
    ocp_nlp_constraints_model_set(config, dims, nlp_in, NN, "lbx", lbxN);
    ocp_nlp_constraints_model_set(config, dims, nlp_in, NN, "ubx", ubxN);

    /************************************************
    * sqp opts
    ************************************************/

    // create opts
    void *nlp_opts = ocp_nlp_solver_opts_create(config, dims);

    // nlp opts
    if (plan->nlp_solver == SQP)
    {

        int max_iter = MAX_SQP_ITERS;
        double tol_stat = 1e-6;
        double tol_eq   = 1e-8;
        double tol_ineq = 1e-8;
        double tol_comp = 1e-8;

        ocp_nlp_solver_opts_set(config, nlp_opts, "max_iter", &max_iter);
        ocp_nlp_solver_opts_set(config, nlp_opts, "tol_stat", &tol_stat);
        ocp_nlp_solver_opts_set(config, nlp_opts, "tol_eq", &tol_eq);
        ocp_nlp_solver_opts_set(config, nlp_opts, "tol_ineq", &tol_ineq);
        ocp_nlp_solver_opts_set(config, nlp_opts, "tol_comp", &tol_comp);
    }
    

    // sim opts
    for (int i = 0; i < NN; ++i)
    {

        if (plan->sim_solver_plan[i].sim_solver == ERK)
        {
            int ns = 4;
            int num_steps = 10;
            ocp_nlp_solver_opts_set_at_stage(config, nlp_opts, i, "dynamics_num_steps", &num_steps);
            ocp_nlp_solver_opts_set_at_stage(config, nlp_opts, i, "dynamics_ns", &ns);
        }
    }

    // partial condensing opts
    if (plan->ocp_qp_solver_plan.qp_solver == PARTIAL_CONDENSING_HPIPM)
    {
        int cond_N = 5;
        ocp_nlp_solver_opts_set(config, nlp_opts, "qp_cond_N", &cond_N);
    }

    /************************************************
    * ocp_nlp_out & solver
    ************************************************/

    ocp_nlp_out *nlp_out = ocp_nlp_out_create(config, dims);

    ocp_nlp_out *sens_nlp_out = ocp_nlp_out_create(config, dims);

    ocp_nlp_solver *solver = ocp_nlp_solver_create(config, dims, nlp_opts, nlp_in);

    /************************************************
    * precomputation (after all options are set)
    ************************************************/

    status = ocp_nlp_precompute(solver, nlp_in, nlp_out);

    /************************************************
    * sqp solve
    ************************************************/

    int timesteps = 40;

    double *x_sim = malloc(nx_*(timesteps+1)*sizeof(double));
    double *u_sim = malloc(nu_*(timesteps+0)*sizeof(double));

    acados_timer timer;
    acados_tic(&timer);

    for (int rep = 0; rep < NREP; rep++)
    {
        // warm start output initial guess of solution
        for (int i=0; i<=NN; i++)
        {
            blasfeo_pack_dvec(2, u0_ref, 1, nlp_out->ux+i, 0);
            blasfeo_pack_dvec(nx[i], x0_ref, 1, nlp_out->ux+i, nu[i]);
        }

        // set x0 as box constraint
        ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "lbx", x0_ref);
        ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "ubx", x0_ref);

        // store x0
        for(int ii=0; ii<nx_; ii++) x_sim[ii] = x0_ref[ii];

        for (int idx = 0; idx < timesteps; idx++)
        {
            // update wind distrurbance as external function parameter
            for (int ii=0; ii<NN; ii++)
            {
                if (plan->sim_solver_plan[ii].sim_solver == ERK)
                {
                    expl_vde_for[ii].set_param(expl_vde_for+ii, wind0_ref+idx+ii);
                }
                else
                {
                    printf("\nWrong sim name\n\n");
                    exit(1);
                }
            }
            // update reference
            for (int i = 0; i <= NN; i++)
            {
                ocp_nlp_cost_model_set(config, dims, nlp_in, i, "yref", &y_ref[(idx + i)*4]);
            }

            // solve NLP
            status = ocp_nlp_solve(solver, nlp_in, nlp_out);

            // evaluate parametric sensitivity of solution
            ocp_nlp_eval_param_sens(solver, "ex", 0, 0, sens_nlp_out);

            // update initial condition
            // TODO(dimitris): maybe simulate system instead of passing x[1] as next state
            ocp_nlp_out_get(config, dims, nlp_out, 1, "x", specific_x);
            ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "lbx", specific_x);
            ocp_nlp_constraints_model_set(config, dims, nlp_in, 0, "ubx", specific_x);

            // store trajectory
            ocp_nlp_out_get(config, dims, nlp_out, 1, "x", x_sim+(idx+1)*nx_);
            ocp_nlp_out_get(config, dims, nlp_out, 0, "u", u_sim+idx*nu_);

            // print info
            if (true)
            {
                int sqp_iter;
                double time_lin, time_qp_sol, time_tot;

                ocp_nlp_get(solver, "sqp_iter", &sqp_iter);
                ocp_nlp_get(solver, "time_tot", &time_tot);
                ocp_nlp_get(solver, "time_qp_sol", &time_qp_sol);
                ocp_nlp_get(solver, "time_lin", &time_lin);

                printf("\nproblem #%d, status %d, iters %d, time (total %f, lin %f, qp_sol %f) ms\n",
                    idx, status, sqp_iter, time_tot*1e3, time_lin*1e3, time_qp_sol*1e3);

                printf("xsim = \n");
                ocp_nlp_out_get(config, dims, nlp_out, 0, "x", x_end);
                d_print_mat(1, nx[0], x_end, 1);
                printf("electrical power = %f\n", 0.944*97/100* x_end[0] * x_end[5]);
            }
            if (status!=0)
            {
                if (plan->nlp_solver == SQP)  // RTI has no residual
                {
                    ocp_nlp_res *residual;
                    ocp_nlp_get(solver, "nlp_res", &residual);
                    printf("\nresiduals\n");
                    ocp_nlp_res_print(dims, residual);
                    exit(1);
                }
            }

            // shift trajectories
            if (true)
            {
                ocp_nlp_out_get(config, dims, nlp_out, NN-1, "u", u_end);
                ocp_nlp_out_get(config, dims, nlp_out, NN-1, "x", x_end);

                shift_states(dims, nlp_out, x_end);
                shift_controls(dims, nlp_out, u_end);
            }

            save_states_to_csv(x_sim, u_sim, timesteps, nx_, nu_);
        }
    }

    double time = acados_toc(&timer)/NREP;

    printf("\n\ntotal time (including printing) = %f ms (time per SQP = %f)\n\n", time*1e3, time*1e3/timesteps);


    /************************************************
    * free memory
    ************************************************/

    external_function_param_casadi_free(expl_vde_for);
   
    free(expl_vde_for);

    ocp_nlp_solver_opts_destroy(nlp_opts);
    ocp_nlp_in_destroy(nlp_in);
    ocp_nlp_out_destroy(nlp_out);
    ocp_nlp_out_destroy(sens_nlp_out);
    ocp_nlp_solver_destroy(solver);
    ocp_nlp_dims_destroy(dims);
    ocp_nlp_config_destroy(config);
    ocp_nlp_plan_destroy(plan);

    free(specific_x);
    free(specific_u);

    free(x_sim);
    free(u_sim);

    free(lZ0);
    free(uZ0);
    free(lz0);
    free(uz0);
    free(lZ1);
    free(uZ1);
    free(lz1);
    free(uz1);
    free(lZN);
    free(uZN);
    free(lzN);
    free(uzN);

    free(W_N);
    free(W);
    free(VxN);
    free(Vx);
    free(Vu);

    free(idxbu0);
    free(lbu0);
    free(ubu0);
    free(idxbx0);
    free(lbx0);
    free(ubx0);

    free(idxbx1);
    free(lbu1);
    free(ubu1);
    free(idxbu1);
    free(lbx1);
    free(ubx1);

    free(idxbxN);
    free(lbxN);
    free(ubxN);

    free(x_end);
    free(u_end);

    /************************************************
    * return
    ************************************************/

    if (status == 0 || (status == 1 && MAX_SQP_ITERS == 1))
        printf("\nsuccess!\n\n");
    else
        printf("\nfailure!\n\n");

    return 0;
}
