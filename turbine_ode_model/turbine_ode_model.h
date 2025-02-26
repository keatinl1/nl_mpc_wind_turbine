
#ifndef turbine_ode_MODEL
#define turbine_ode_MODEL

#ifdef __cplusplus
extern "C" {
#endif


  
// implicit ODE: function
int turbine_ode_impl_dae_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int turbine_ode_impl_dae_fun_work(int *, int *, int *, int *);
const int *turbine_ode_impl_dae_fun_sparsity_in(int);
const int *turbine_ode_impl_dae_fun_sparsity_out(int);
int turbine_ode_impl_dae_fun_n_in(void);
int turbine_ode_impl_dae_fun_n_out(void);

// implicit ODE: function + jacobians
int turbine_ode_impl_dae_fun_jac_x_xdot_z(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int turbine_ode_impl_dae_fun_jac_x_xdot_z_work(int *, int *, int *, int *);
const int *turbine_ode_impl_dae_fun_jac_x_xdot_z_sparsity_in(int);
const int *turbine_ode_impl_dae_fun_jac_x_xdot_z_sparsity_out(int);
int turbine_ode_impl_dae_fun_jac_x_xdot_z_n_in(void);
int turbine_ode_impl_dae_fun_jac_x_xdot_z_n_out(void);

// implicit ODE: jacobians only
int turbine_ode_impl_dae_jac_x_xdot_u_z(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int turbine_ode_impl_dae_jac_x_xdot_u_z_work(int *, int *, int *, int *);
const int *turbine_ode_impl_dae_jac_x_xdot_u_z_sparsity_in(int);
const int *turbine_ode_impl_dae_jac_x_xdot_u_z_sparsity_out(int);
int turbine_ode_impl_dae_jac_x_xdot_u_z_n_in(void);
int turbine_ode_impl_dae_jac_x_xdot_u_z_n_out(void);

// implicit ODE - for lifted_irk
int turbine_ode_impl_dae_fun_jac_x_xdot_u(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int turbine_ode_impl_dae_fun_jac_x_xdot_u_work(int *, int *, int *, int *);
const int *turbine_ode_impl_dae_fun_jac_x_xdot_u_sparsity_in(int);
const int *turbine_ode_impl_dae_fun_jac_x_xdot_u_sparsity_out(int);
int turbine_ode_impl_dae_fun_jac_x_xdot_u_n_in(void);
int turbine_ode_impl_dae_fun_jac_x_xdot_u_n_out(void);
  



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // turbine_ode_MODEL
