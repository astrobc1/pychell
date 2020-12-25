import streamlit as st
import numpy as np


def data_input(data):
    # Use or ignore any instruments as st checkboxes.
    for inst in list(data.keys()):
        use_data = st.checkbox(label=inst + " " + str(len(data[inst].t)), value=True)
        if not use_data:
            del data[inst]
            
            
def rv_action_inputs(pars, planets_dict, optprob):
    comps = {}
    st.markdown('## Actions')
    comps["optim_button"] = st.button(label='Run Max Like')
    comps["sample_button"] = st.button(label='Run MCMC')
    comps["model_comp_button"] = st.button(label='Model Comparison')
    comps["period_search_button"] = st.button('Period Search')
    # Kind of period search
    st.markdown('## Period Search Options:')
    st.markdown('### Periodogram Type:')
    comps["persearch_kind_input"] = st.radio(label="", options=["GLS", "Brute Force"])
    remove_planets_inputs = {}
    st.markdown('### Remove Planets:')
    remove_planets_pgram = []
    for planet in planets_dict:
        remove_planets_inputs[planet] = st.checkbox(label=str(planet))
        if remove_planets_inputs[planet]:
            remove_planets_pgram.append(planet)
    
    comps["remove_planet_inputs"] = remove_planets_pgram
    pgcols = st.beta_columns(2)
    comps["period_min_input"] = pgcols[0].text_input(label='Period min', value=1.1)
    comps["period_max_input"] = pgcols[1].text_input(label='Period max', value=100)
    
    return comps