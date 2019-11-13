"""
croston model for intermittent time series
"""

import numpy
import pandas
from scipy.optimize import minimize

def fit_croston(
                    data_name,
                    model_time_series_required_length,
                    input_endog,
                    input_dates,
                    input_length,
                    forecast_length,
                    time_grain,
                    input_endog_shifted,
                    forecast_shifted_response,
                    error_logger,
                    training_length_in_years,
                    time_series_class,
                    holidays,
                    training_exog_var,
                    forecast_exog_var
                ):
        """

        :param data_name:
        :param model_time_series_required_length:
        :param input_endog:
        :param input_dates:
        :param input_length:
        :param forecast_length:
        :param time_grain:
        :param input_endog_shifted:
        :param forecast_shifted_response:
        :param error_logger:
        :param training_length_in_years:
        :param time_series_class:
        :param holidays:
        :param training_exog_var:
        :param forecast_exog_var:
        :return:
        """
        model = 'croston'
        epsilon = 1e-7
        input_series = numpy.asarray(input_endog.values)
        nzd = numpy.where(input_series != 0)[0]
        
        if time_series_class == 'intermittent' and training_length_in_years > model_time_series_required_length.get(model, 0.5) and list(nzd) != [0]:
                
                try:
                    w_opt = _croston_opt(
                                            input_series = numpy.asarray(input_endog.values),
                                            input_series_length = input_length,
                                            epsilon = epsilon,                                            
                                            w = None,
                                            nop = 1,
                                            croston_variant = 'original'
                                        )
                    
                    croston_training_result = _croston(
                                                            input_series = numpy.asarray(input_endog.values), 
                                                            input_series_length = input_length,
                                                            w = w_opt, 
                                                            h = forecast_length,
                                                            epsilon = epsilon,
                                                            croston_variant = 'original'
                                                      )
                    croston_model = croston_training_result['model']
                    croston_fittedvalues = croston_training_result['in_sample_forecast']
                    
                    croston_forecast = croston_training_result['out_of_sample_forecast']

                except Exception as e:
                    
                    croston_model = None
                    croston_fittedvalues = None
                    croston_forecast = None
                    error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
        
        else:
            
            croston_model = None
            croston_fittedvalues = None
            croston_forecast = None        
        
        
        return croston_model, croston_fittedvalues, croston_forecast

def _croston(
                 input_series, 
                 input_series_length,
                 w, 
                 h,                  
                 epsilon,
                 croston_variant = 'original'
             ):
    """

    :param input_series:
    :param input_series_length:
    :param w:
    :param h:
    :param epsilon:
    :param croston_variant:
    :return:
    """
    # Croston decomposition
    nzd = numpy.where(input_series != 0)[0] # find location of non-zero demand
    
    k = len(nzd)
    z = input_series[nzd] # demand
    
    x = numpy.concatenate([[nzd[0]], numpy.diff(nzd)]) # intervals

    # initialize
    
    init = [z[0], numpy.mean(x)]
    
    zfit = numpy.array([None] * k)
    xfit = numpy.array([None] * k)

    # assign initial values and prameters
    
    zfit[0] = init[0]
    xfit[0] = init[1]

    if len(w) == 1:
        a_demand = w[0]
        a_interval = w[0]
    
    else:
        a_demand = w[0]
        a_interval = w[1]
    
    # compute croston variant correction factors
    #   sba: syntetos-boylan approximation
    #   sbj: shale-boylan-johnston
    #   tsb: teunter-syntetos-babai        
    
    if croston_variant == 'sba':
        correction_factor = 1 - (a_interval / 2)
    
    elif croston_variant == 'sbj':
        correction_factor = (1 - a_interval / (2 - a_interval + epsilon))
        
    else:
        correction_factor = 1
    
    # fit model
    
    for i in range(1,k):
        zfit[i] = zfit[i-1] + a_demand * (z[i] - zfit[i-1]) # demand
        xfit[i] = xfit[i-1] + a_interval * (x[i] - xfit[i-1]) # interval
        
    cc = correction_factor * zfit / (xfit + epsilon)
    
    croston_model = {
                        'a_demand':             a_demand,
                        'a_interval':           a_interval,
                        'demand_series':        pandas.Series(zfit),
                        'interval_series':      pandas.Series(xfit),
                        'demand_process':       pandas.Series(cc),
                        'correction_factor':    correction_factor
                    }
    
    # calculate in-sample demand rate
    
    frc_in = numpy.zeros(input_series_length)
    tv = numpy.concatenate([nzd, [input_series_length]]) # Time vector used to create frc_in forecasts
    
    for i in range(k):
        frc_in[tv[i]:min(tv[i+1], input_series_length)] = cc[i]

    # forecast out_of_sample demand rate
    
    if h > 0:
        frc_out = numpy.array([cc[k-1]] * h)
        
    else:
        frc_out = None
    
    return_dictionary = {
                            'model':                    croston_model,
                            'in_sample_forecast':       pandas.Series(frc_in),
                            'out_of_sample_forecast':   pandas.Series(frc_out)
                        }
    
    return return_dictionary

def _croston_opt(
                    input_series, 
                    input_series_length, 
                    epsilon,
                    w = None,
                    nop = 1,
                    croston_variant = 'original'
                ):
    """

    :param input_series:
    :param input_series_length:
    :param epsilon:
    :param w:
    :param nop:
    :param croston_variant:
    :return:
    """
    p0 = numpy.array([0.1] * nop)
            
    wopt = minimize(
                        fun = _croston_cost, 
                        x0 = p0, 
                        method='Nelder-Mead',
                        args=(input_series, input_series_length, epsilon, croston_variant)
                    )
    
    constrained_wopt = numpy.minimum([1], numpy.maximum([0], wopt.x))   
    
    return constrained_wopt
    

def _croston_cost(
                    p0,
                    input_series,
                    input_series_length,
                    epsilon,
                    croston_variant
                ):
    """

    :param p0:
    :param input_series:
    :param input_series_length:
    :param epsilon:
    :param croston_variant:
    :return:
    """
    # cost function for croston and variants
    
    frc_in = _croston(input_series = input_series, input_series_length = input_series_length, w=p0, h=0, epsilon = epsilon, croston_variant = croston_variant)['in_sample_forecast']
        
    E = input_series - frc_in
    E = E[E != numpy.array(None)]
    E = numpy.mean(E ** 2)

    return E
