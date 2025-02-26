// standard
#include <stdio.h>
#include <stdlib.h>

// acados
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_turbine_ode.h"


#define NX     TURBINE_ODE_NX
#define NP     TURBINE_ODE_NP
#define NU     TURBINE_ODE_NU
#define NBX0   TURBINE_ODE_NBX0


int main()
{
    turbine_ode_solver_capsule *acados_ocp_capsule = turbine_ode_acados_create_capsule();

    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = TURBINE_ODE_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;

    int status = turbine_ode_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("turbine_ode_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = turbine_ode_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = turbine_ode_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = turbine_ode_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = turbine_ode_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = turbine_ode_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = turbine_ode_acados_get_nlp_opts(acados_ocp_capsule);

    // initial condition
    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = 0;
    ubx0[0] = 0;
    lbx0[1] = 0.1 * 3.141592653589793;
    ubx0[1] = 0.1 * 3.141592653589793;
    lbx0[2] = 0;
    ubx0[2] = 0;
    lbx0[3] = 0;
    ubx0[3] = 0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;
    x_init[3] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];

    // solve ocp in loop
    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        status = turbine_ode_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\nsolved ocp %d times, solution not printed.\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("turbine_ode_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("turbine_ode_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);

    turbine_ode_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);

    // free solver
    status = turbine_ode_acados_free(acados_ocp_capsule);
    if (status) {
        printf("turbine_ode_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = turbine_ode_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("turbine_ode_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
