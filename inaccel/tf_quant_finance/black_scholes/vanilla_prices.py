# Lint as: python3
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Black Scholes prices of a batch of European options."""

import inaccel.coral as inaccel
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.black_scholes import option_price as option_price_ref
from tf_quant_finance.black_scholes import binary_price as binary_price_ref


def ndinaccel(shape, dtype):
  with inaccel.allocator:
    return np.ndarray(shape, dtype)


def asinaccel(obj, dtype=None):
  with inaccel.allocator:
    return np.array(obj, dtype, copy=not inaccel.allocator.handles(obj), ndmin=1)


def option_price(*,
                 volatilities,
                 strikes,
                 expiries,
                 spots=None,
                 forwards=None,
                 discount_rates=None,
                 continuous_dividends=None,
                 cost_of_carries=None,
                 discount_factors=None,
                 is_call_options=None,
                 is_normal_volatility=False,
                 dtype=None,
                 name=None):
  """Computes the Black Scholes price for a batch of call or put options.

  #### Example

  ```python
    # Price a batch of 5 vanilla call options.
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Strikes will automatically be broadcasted to shape [5].
    strikes = np.array([3.0])
    # Expiries will be broadcast to shape [5], i.e. each option has strike=3
    # and expiry = 1.
    expiries = 1.0
    computed_prices = tff.black_scholes.option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards)
  # Expected print output of computed prices:
  # [ 0.          2.          2.04806848  1.00020297  2.07303131]
  ```

  #### References:
  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Black-Scholes model. Available at:
    https://en.wikipedia.org/w/index.php?title=Black%E2%80%93Scholes_model

  Args:
    volatilities: Real `Tensor` of any shape and dtype. The volatilities to
      expiry of the options to price.
    strikes: A real `Tensor` of the same dtype and compatible shape as
      `volatilities`. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
      `volatilities`. The expiry of each option. The units should be such that
      `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `volatilities`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `volatilities`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    discount_rates: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates. At most one of
      discount_rates and discount_factors can be supplied.
      Default value: `None`, equivalent to r = 0 and discount factors = 1 when
      discount_factors also not given.
    continuous_dividends: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      If not `None`, `cost_of_carries` is calculated as r - q,
      where r are the `discount_rates` and q is `continuous_dividends`. Either
      this or `cost_of_carries` can be given.
      Default value: `None`, equivalent to q = 0.
    cost_of_carries: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      Cost of storing a physical commodity, the cost of interest paid when
      long, or the opportunity cost, or the cost of paying dividends when short.
      If not `None`, and `spots` is supplied, used to calculate forwards from
      `spots`: F = e^(bT) * S, where F is the forwards price, b is the cost of
      carries, T is expiries and S is the spot price. If `None`, value assumed
      to be equal to the `discount_rate` - `continuous_dividends`
      Default value: `None`, equivalent to b = r.
    discount_factors: An optional real `Tensor` of same dtype as the
      `volatilities`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with discount_rate and cost_of_carry.
      If neither is given, no discounting is applied (i.e. the undiscounted
      option price is returned). If `spots` is supplied and `discount_factors`
      is not `None` then this is also used to compute the forwards to expiry.
      At most one of discount_rates and discount_factors can be supplied.
      Default value: `None`, which maps to e^(-rT) calculated from
      discount_rates.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    is_normal_volatility: An optional Python boolean specifying whether the
      `volatilities` correspond to lognormal Black volatility (if False) or
      normal Black volatility (if True).
      Default value: False, which corresponds to lognormal volatility.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: `None` which maps to the default dtype inferred by
        TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: `None` which is mapped to the default name `option_price`.

  Returns:
    option_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
    ValueError: If both `discount_rates` and `discount_factors` is supplied.
    ValueError: If both `continuous_dividends` and `cost_of_carries` is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if (discount_rates is not None) and (discount_factors is not None):
    raise ValueError('At most one of discount_rates and discount_factors may '
                     'be supplied')
  if (continuous_dividends is not None) and (cost_of_carries is not None):
    raise ValueError('At most one of continuous_dividends and cost_of_carries '
                     'may be supplied')

  if is_normal_volatility:
    return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  if (dtype is not None) and (np.dtype(dtype) != np.float32):
    return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  size = np.int32(1)

  volatilities = asinaccel(volatilities, np.float32)
  size = np.maximum(size, volatilities.size, dtype=np.int32)

  strikes = asinaccel(strikes, np.float32)
  size = np.maximum(size, strikes.size, dtype=np.int32)

  expiries = asinaccel(expiries, np.float32)
  size = np.maximum(size, expiries.size, dtype=np.int32)

  if forwards is not None:
    forwards = asinaccel(forwards, np.float32)
    size = np.maximum(size, forwards.size, dtype=np.int32)
  else:
    spots = asinaccel(spots, np.float32)
    size = np.maximum(size, spots.size, dtype=np.int32)

  if discount_factors is not None:
    discount_factors = asinaccel(discount_factors, np.float32)
    size = np.maximum(size, discount_factors.size, dtype=np.int32)
  elif discount_rates is not None:
    discount_rates = asinaccel(discount_rates, np.float32)
    size = np.maximum(size, discount_rates.size, dtype=np.int32)

  if cost_of_carries is not None:
    cost_of_carries = asinaccel(cost_of_carries, np.float32)
    size = np.maximum(size, cost_of_carries.size, dtype=np.int32)
  elif continuous_dividends is not None:
    continuous_dividends = asinaccel(continuous_dividends, np.float32)
    size = np.maximum(size, continuous_dividends.size, dtype=np.int32)

  if is_call_options is not None:
    is_call_options = asinaccel(is_call_options, np.bool_)
    size = np.maximum(size, is_call_options.size, dtype=np.int32)

  if volatilities.size == 1:
    tmp = volatilities[0]
    volatilities = ndinaccel(size, np.float32)
    volatilities.fill(tmp)
  elif volatilities.size != size:
    return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  if strikes.size == 1:
    tmp = strikes[0]
    strikes = ndinaccel(size, np.float32)
    strikes.fill(tmp)
  elif strikes.size != size:
    return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  if expiries.size == 1:
    tmp = expiries[0]
    expiries = ndinaccel(size, np.float32)
    expiries.fill(tmp)
  elif expiries.size != size:
    return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  if forwards is not None:
    if forwards.size == 1:
      tmp = forwards[0]
      forwards = ndinaccel(size, np.float32)
      forwards.fill(tmp)
    elif not forwards.size == size:
      return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)
  else:
    if spots.size == 1:
      tmp = spots[0]
      spots = ndinaccel(size, np.float32)
      spots.fill(tmp)
    elif not spots.size == size:
      return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  if cost_of_carries is None:
    if discount_rates is None:
      if discount_factors is not None:
        discount_rates = -np.log(discount_factors) / expiries
      else:
        discount_rates = np.ndarray(size, np.float32)
        discount_rates.fill(0.0)

    if continuous_dividends is None:
      continuous_dividends = np.ndarray(size, np.float32)
      continuous_dividends.fill(0.0)

    cost_of_carries = asinaccel(discount_rates - continuous_dividends, np.float32)
  else:
    if cost_of_carries.size == 1:
      tmp = cost_of_carries[0]
      cost_of_carries = ndinaccel(size, np.float32)
      cost_of_carries.fill(tmp)
    elif not cost_of_carries.size == size:
      return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  if discount_factors is None:
    if discount_rates is not None:
      discount_factors = asinaccel(np.exp(discount_rates * expiries), np.float32)
    else:
      discount_factors = ndinaccel(size, np.float32)
      discount_factors.fill(1.0)
  else:
    if discount_factors.size == 1:
      tmp = discount_factors[0]
      discount_factors = ndinaccel(size, np.float32)
      discount_factors.fill(tmp)
    elif not discount_factors.size == size:
      return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  if is_call_options is None:
    is_call_options = ndinaccel(size, np.bool_)
    is_call_options.fill(True)
  else:
    if is_call_options.size == 1:
      tmp = is_call_options[0]
      is_call_options = ndinaccel(size, np.bool_)
      is_call_options.fill(tmp)
    elif is_call_options.size != size:
      return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)

  option_price = ndinaccel(size, np.float32)

  try:
    request = inaccel.request('com.inaccel.quantitativeFinance.blackScholes.option-price')

    if (forwards is not None):
      request.arg(forwards)
      request.arg(np.int32(False))
    else:
      request.arg(spots)
      request.arg(np.int32(True))
    request.arg(cost_of_carries)
    request.arg(strikes)
    request.arg(expiries)
    request.arg(discount_factors)
    request.arg(volatilities)
    request.arg(is_call_options)
    request.arg(size)
    request.arg(option_price)

    inaccel.submit(request).result()

    return tf.convert_to_tensor(option_price, name=(name or 'option_price'))
  except:
    return option_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_rates=discount_rates, continuous_dividends=continuous_dividends, cost_of_carries=cost_of_carries, discount_factors=discount_factors, is_call_options=is_call_options, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name)


# TODO(b/154806390): Binary price signature should be the same as that of the
# vanilla price.
def binary_price(*,
                 volatilities,
                 strikes,
                 expiries,
                 spots=None,
                 forwards=None,
                 discount_factors=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
  """Computes the Black Scholes price for a batch of binary call or put options.

  The binary call (resp. put) option priced here is that which pays off a unit
  of cash if the underlying asset has a value greater (resp. smaller) than the
  strike price at expiry. Hence the binary option price is the discounted
  probability that the asset will end up higher (resp. lower) than the
  strike price at expiry.

  #### Example

  ```python
    # Price a batch of 5 binary call options.
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Strikes will automatically be broadcasted to shape [5].
    strikes = np.array([3.0])
    # Expiries will be broadcast to shape [5], i.e. each option has strike=3
    # and expiry = 1.
    expiries = 1.0
    computed_prices = tff.black_scholes.binary_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards)
  # Expected print output of prices:
  # [0.         0.         0.15865525 0.99764937 0.85927418]
  ```

  #### References:

  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Binary option. Available at:
  https://en.wikipedia.org/w/index.php?title=Binary_option

  Args:
    volatilities: Real `Tensor` of any shape and dtype. The volatilities to
      expiry of the options to price.
    strikes: A real `Tensor` of the same dtype and compatible shape as
      `volatilities`. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
      `volatilities`. The expiry of each option. The units should be such that
      `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `volatilities`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `volatilities`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    discount_factors: An optional real `Tensor` of same dtype as the
      `volatilities`. If not None, these are the discount factors to expiry
      (i.e. e^(-rT)). If None, no discounting is applied (i.e. the undiscounted
      option price is returned). If `spots` is supplied and `discount_factors`
      is not None then this is also used to compute the forwards to expiry.
      Default value: None, equivalent to discount factors = 1.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by TensorFlow
        (float32).
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name `binary_price`.

  Returns:
    binary_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the binary options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')

  if (dtype is not None) and (np.dtype(dtype) != np.float32):
    return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)

  size = np.int32(1)

  volatilities = asinaccel(volatilities, np.float32)
  size = np.maximum(size, volatilities.size, dtype=np.int32)

  strikes = asinaccel(strikes, np.float32)
  size = np.maximum(size, strikes.size, dtype=np.int32)

  expiries = asinaccel(expiries, np.float32)
  size = np.maximum(size, expiries.size, dtype=np.int32)

  if forwards is not None:
    forwards = asinaccel(forwards, np.float32)
    size = np.maximum(size, forwards.size, dtype=np.int32)
  else:
    spots = asinaccel(spots, np.float32)
    size = np.maximum(size, spots.size, dtype=np.int32)

  if discount_factors is not None:
    discount_factors = asinaccel(discount_factors, np.float32)
    size = np.maximum(size, discount_factors.size, dtype=np.int32)

  if is_call_options is not None:
    is_call_options = asinaccel(is_call_options, np.bool_)
    size = np.maximum(size, is_call_options.size, dtype=np.int32)

  if volatilities.size == 1:
    tmp = volatilities[0]
    volatilities = ndinaccel(size, np.float32)
    volatilities.fill(tmp)
  elif volatilities.size != size:
    return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)

  if strikes.size == 1:
    tmp = strikes[0]
    strikes = ndinaccel(size, np.float32)
    strikes.fill(tmp)
  elif strikes.size != size:
    return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)

  if expiries.size == 1:
    tmp = expiries[0]
    expiries = ndinaccel(size, np.float32)
    expiries.fill(tmp)
  elif expiries.size != size:
    return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)

  if forwards is not None:
    if forwards.size == 1:
      tmp = forwards[0]
      forwards = ndinaccel(size, np.float32)
      forwards.fill(tmp)
    elif forwards.size != size:
      return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)
  else:
    if spots.size == 1:
      tmp = spots[0]
      spots = ndinaccel(size, np.float32)
      spots.fill(tmp)
    elif spots.size != size:
      return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)

  if discount_factors is None:
    discount_factors = ndinaccel(size, np.float32)
    discount_factors.fill(1.0)
  else:
    if discount_factors.size == 1:
      tmp = discount_factors[0]
      discount_factors = ndinaccel(size, np.float32)
      discount_factors.fill(tmp)
    elif discount_factors.size != size:
      return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)

  if is_call_options is None:
    is_call_options = ndinaccel(size, np.bool_)
    is_call_options.fill(True)
  else:
    if is_call_options.size == 1:
      tmp = is_call_options[0]
      is_call_options = ndinaccel(size, np.bool_)
      is_call_options.fill(tmp)
    elif is_call_options.size != size:
      return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)

  binary_price = ndinaccel(size, np.float32)

  try:
    request = inaccel.request('com.inaccel.quantitativeFinance.blackScholes.binary-price')

    if (forwards is not None):
      request.arg(forwards)
      request.arg(np.int32(False))
    else:
      request.arg(spots)
      request.arg(np.int32(True))
    request.arg(strikes)
    request.arg(expiries)
    request.arg(discount_factors)
    request.arg(volatilities)
    request.arg(is_call_options)
    request.arg(size)
    request.arg(binary_price)

    inaccel.submit(request).result()

    return tf.convert_to_tensor(binary_price, name=(name or 'binary_price'))
  except:
    return binary_price_ref(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype, name=name)
