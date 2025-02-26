# define sources and use make's implicit rules to generate object files (*.o)

# model
MODEL_SRC=
MODEL_SRC+= turbine_ode_model/turbine_ode_impl_dae_fun.c
MODEL_SRC+= turbine_ode_model/turbine_ode_impl_dae_fun_jac_x_xdot_z.c
MODEL_SRC+= turbine_ode_model/turbine_ode_impl_dae_jac_x_xdot_u_z.c
MODEL_OBJ := $(MODEL_SRC:.c=.o)
# optimal control problem - mostly CasADi exports
OCP_SRC=
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_0_fun.c
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_0_fun_jac_ut_xt.c
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_0_hess.c
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_fun.c
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_fun_jac_ut_xt.c
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_hess.c
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_e_fun.c
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_e_fun_jac_ut_xt.c
OCP_SRC+= turbine_ode_cost/turbine_ode_cost_y_e_hess.c

OCP_SRC+= acados_solver_turbine_ode.c
OCP_OBJ := $(OCP_SRC:.c=.o)

# for target example
EX_SRC= main_turbine_ode.c
EX_OBJ := $(EX_SRC:.c=.o)
EX_EXE := $(EX_SRC:.c=)


# combine model, (potentially) sim and ocp object files
OBJ=
OBJ+= $(MODEL_OBJ)
OBJ+= $(OCP_OBJ)

EXTERNAL_DIR=
EXTERNAL_LIB=

INCLUDE_PATH = /home/luke/acados/include
LIB_PATH = /home/luke/acados/lib

# preprocessor flags for make's implicit rules
CPPFLAGS+= -I$(INCLUDE_PATH)
CPPFLAGS+= -I$(INCLUDE_PATH)/acados
CPPFLAGS+= -I$(INCLUDE_PATH)/blasfeo/include
CPPFLAGS+= -I$(INCLUDE_PATH)/hpipm/include


# define the c-compiler flags for make's implicit rules
CFLAGS = -fPIC -std=c99   -O2#-fno-diagnostics-show-line-numbers -g


# linker flags
LDFLAGS+= -L$(LIB_PATH)


# link to libraries
LDLIBS+= -lacados
LDLIBS+= -lhpipm
LDLIBS+= -lblasfeo
LDLIBS+= -lqpOASES_e   # Add this line
LDLIBS+= -lm


# libraries
LIBACADOS_SOLVER=libacados_solver_turbine_ode.so
LIBACADOS_OCP_SOLVER=libacados_ocp_solver_turbine_ode.so
LIBACADOS_SIM_SOLVER=lib$(SIM_SRC:.c=.so)

# virtual targets
.PHONY : all clean

all: clean example
shared_lib: bundled_shared_lib ocp_shared_lib sim_shared_lib

# some linker targets
example: $(EX_OBJ) $(OBJ)
	$(CC) $^ -o $(EX_EXE) $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) $(OBJ) $(EX_OBJ) $(EX_SIM_OBJ)
	$(RM) $(LIBACADOS_SOLVER) $(LIBACADOS_OCP_SOLVER) $(LIBACADOS_SIM_SOLVER)
	$(RM) $(EX_EXE) $(EX_SIM_EXE)
