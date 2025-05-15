import jax
import jax.numpy as jnp


from .config import Constants, FMConfig
from .fourier import my_fft, my_ifft

def get_delta_in(input_arr, cte: Constants, fm_cfg: FMConfig):
    delta = my_fft(input_arr, cte.L3) * jnp.sqrt(fm_cfg.pow_spec * cte.INV_R3)
    return delta 

def gen_input_arr(key, cte: Constants):
    input_arr = jax.random.normal(key, shape=(cte.N,) * 3)
    return input_arr

