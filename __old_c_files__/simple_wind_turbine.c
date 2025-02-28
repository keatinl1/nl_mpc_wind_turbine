// external
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// acados
#include "acados/utils/external_function_generic.h"
#include "acados_c/external_function_interface.h"
#include "acados_c/sim_interface.h"

// wt model
#include "examples/c/wt_model_nx3/wt_model.h"

// x0 and u for simulation
#include "examples/c/wt_model_nx3/u_x0.c"

// blasfeo
#include "blasfeo_common.h"
#include "blasfeo_d_aux.h"
#include "blasfeo_d_aux_ext_dep.h"
#include "blasfeo_d_blas.h"
#include "blasfeo_v_aux_ext_dep.h"


int main()
{

/************************************************
* initialize common stuff (for all integrators)
************************************************/

    int nx = 3;
    int nu = 4;
	int nz = 0;
    int NF = nx + nu; // columns of forward seed

    double T = 0.05; // simulation time

	double x_sim[nx*(nsim+1)];
	for (int ii = 0; ii < nx; ii++)
		x_sim[ii] = x0[ii];

	/************************************************
	* external functions (explicit model)
	************************************************/
    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);
    ext_fun_opts.external_workspace = true;

	// expl_ode_fun
	external_function_casadi expl_ode_fun;
	expl_ode_fun.casadi_fun = &casadi_expl_ode_fun;
	expl_ode_fun.casadi_work = &casadi_expl_ode_fun_work;
	expl_ode_fun.casadi_sparsity_in = &casadi_expl_ode_fun_sparsity_in;
	expl_ode_fun.casadi_sparsity_out = &casadi_expl_ode_fun_sparsity_out;
	expl_ode_fun.casadi_n_in = &casadi_expl_ode_fun_n_in;
	expl_ode_fun.casadi_n_out = &casadi_expl_ode_fun_n_out;
	external_function_casadi_create(&expl_ode_fun, &ext_fun_opts);

	// expl_vde_for
	external_function_casadi expl_vde_for;
	expl_vde_for.casadi_fun = &casadi_expl_vde_for;
	expl_vde_for.casadi_work = &casadi_expl_vde_for_work;
	expl_vde_for.casadi_sparsity_in = &casadi_expl_vde_for_sparsity_in;
	expl_vde_for.casadi_sparsity_out = &casadi_expl_vde_for_sparsity_out;
	expl_vde_for.casadi_n_in = &casadi_expl_vde_for_n_in;
	expl_vde_for.casadi_n_out = &casadi_expl_vde_for_n_out;
	external_function_casadi_create(&expl_vde_for, &ext_fun_opts);

	// expl_vde_adj
	external_function_casadi expl_vde_adj;
	expl_vde_adj.casadi_fun = &casadi_expl_vde_adj;
	expl_vde_adj.casadi_work = &casadi_expl_vde_adj_work;
	expl_vde_adj.casadi_sparsity_in = &casadi_expl_vde_adj_sparsity_in;
	expl_vde_adj.casadi_sparsity_out = &casadi_expl_vde_adj_sparsity_out;
	expl_vde_adj.casadi_n_in = &casadi_expl_vde_adj_n_in;
	expl_vde_adj.casadi_n_out = &casadi_expl_vde_adj_n_out;
	external_function_casadi_create(&expl_vde_adj, &ext_fun_opts);

	int nss = 0;

    /************************************************
    * sim plan & config
    ************************************************/
    sim_solver_plan_t plan;
    plan.sim_solver = ERK;

    // create correct config based on plan
    sim_config *config = sim_config_create(plan);

    /************************************************
    * sim dims
    ************************************************/

    void *dims = sim_dims_create(config);
    
    sim_dims_set(config, dims, "nx", &nx);
    sim_dims_set(config, dims, "nu", &nu);
    sim_dims_set(config, dims, "nz", &nz);

    /************************************************
    * sim opts
    ************************************************/

    sim_opts *opts = sim_opts_create(config, dims);

    // opts->ns = 4; // number of stages in rk integrator
    // opts->num_steps = 3; // number of integration steps

    opts->sens_forw = true;
    opts->sens_adj = true;

    // ERK
    opts->ns = 4; // number of stages in rk integrator
    opts->sens_adj = false;


    /************************************************
    * sim in / out
    ************************************************/

    sim_in *in = sim_in_create(config, dims);
    sim_out *out = sim_out_create(config, dims);

    in->T = T;

    // external functions
    config->model_set(in->model, "expl_ode_fun", &expl_ode_fun);
    config->model_set(in->model, "expl_vde_for", &expl_vde_for);
    config->model_set(in->model, "expl_vde_adj", &expl_vde_adj);
 
    /************************************************
    * sim solver
    ************************************************/

    sim_solver *sim_solver = sim_solver_create(config, dims, opts, in);

    int acados_return;

    sim_precompute(sim_solver, in, out);

    acados_timer timer;
    acados_tic(&timer);

    int nsim0 = 1;//nsim;

    double cpu_time = 0.0;
    double la_time = 0.0;
    double ad_time = 0.0;

    printf("\n---> testing integrator %d (num_steps = %d, num_stages = %d, jac_reuse = %d, newton_iter = %d )\n",
                nss, opts->num_steps, opts->ns, opts->jac_reuse, opts->newton_iter);

    for (int ii = 0; ii < nsim0; ii++)
    {
        // x
        for (int jj = 0; jj < nx; jj++)
            in->x[jj] = x_sim[ii*nx+jj];

        // u
        for (int jj = 0; jj < nu; jj++)
            in->u[jj] = u_sim[ii*nu+jj];

        acados_return = sim_solve(sim_solver, in, out);
        if (acados_return != 0)
            printf("error in sim solver\n");

        cpu_time += out->info->CPUtime;
        la_time += out->info->LAtime;
        ad_time += out->info->ADtime;

        // x_out
        for (int jj = 0; jj < nx; jj++)
            x_sim[(ii+1)*nx+jj] = out->xn[jj];

    }
    double total_cpu_time = acados_toc(&timer);





    /************************************************
    * printing
    ************************************************/

    printf("\nxn: \n");
    for (int ii = 0; ii < nx; ii++)
        printf("%8.5f ", x_sim[nsim0*nx+ii]);
    printf("\n\n");

    printf("time for %d simulation steps: %f ms (AD time: %f ms (%5.2f%%))\n\n", 
        nsim0, 1e3*total_cpu_time, 1e3*ad_time, 1e2*ad_time/cpu_time);

    /************************************************
	* free up everything
	************************************************/

    free(sim_solver);
    free(in);
    free(out);

    free(opts);
    free(config);
    free(dims);
	
    external_function_casadi_free(&expl_ode_fun);
	external_function_casadi_free(&expl_vde_for);
	external_function_casadi_free(&expl_vde_adj);

	/************************************************
	* return
	************************************************/

	printf("\nsuccess! (RESULT NOT CHECKED) \n\n");

    return 0;
}
