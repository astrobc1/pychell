import streamlit as st
import numpy as np
import importlib
import pychell.utils as pcutils
import pandas as pd
import plotly

def make_title(title):
    return st.title(title)

class StreamlitComponent:
    
    def __init__(self, comps, label="", *args, **kwargs):
        self.comps = comps
        self.label = label
        
    def write(self, *args, **kwargs):
        pass

class DataSelector(StreamlitComponent):
    
    def __init__(self, comps, data):
        super().__init__(comps=comps, label="Data Selector")
        self.data = data
        self.write()
        
    def write(self):
        # Use or ignore any instruments as st checkboxes.
        st.markdown("## Data")
        for data in self.data.values():
            self.comps["use_" + data.label] = st.checkbox(label=data.label + " [" + str(len(data.t)) + "]", value=True)
        self.update_data()
        return self.comps
    
    def update_data(self):
        for inst in list(self.data.keys()):
            if not self.comps["use_" + inst]:
                del self.data[inst]
    
class RVActions(StreamlitComponent):
    
    def __init__(self, comps, optprob, has_gp=False):
        super().__init__(comps=comps, label="RV Action")
        self.has_gp = has_gp
        self.optprob = optprob
        self.write()
        
    def write(self):
        
        # Primary actions
        st.markdown('## Actions')
        self.comps["max_like_button"] = st.button(label='Max Like')
        self.comps["sample_button"] = st.button(label='MCMC')
        self.comps["model_comp_button"] = st.button(label='Model Comparison')
        self.comps["per_search_button"] = st.button('Period Search')
        if self.has_gp:
            self.comps["rvcolor_button"] = st.button('RV Color')
            self.comps["use_gp_input"] = st.checkbox(label="Use GP", value=True)
        
        # Period search options
        st.markdown('## Period Search Options:')
        st.markdown('### Periodogram Type:')
        self.comps["persearch_kind_input"] = st.radio(label="", options=["GLS", "Brute Force"])
        self.comps["persearch_remove"] = {}
        st.markdown('### Remove Planets:')
        remove_planets_pgram = []
        for planet_index in self.optprob.planets_dict:
            self.comps["persearch_remove"][planet_index] = st.checkbox(label=str(planet_index))
            if self.comps["persearch_remove"][planet_index]:
                remove_planets_pgram.append(planet_index)
    
        self.comps["remove_planet_inputs"] = remove_planets_pgram
        pgcols = st.beta_columns(2)
        self.comps["persearch_min_input"] = pgcols[0].text_input(label='Period min', value=1.1)
        self.comps["persearch_max_input"] = pgcols[1].text_input(label='Period max', value=100)
        return self.comps
    
class MaxLikeResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, maxlike_result):
        super().__init__(comps=comps, label="Optimize Result")
        self.optprob = optprob
        self.maxlike_result = maxlike_result
        self.write()
        
    def write(self):
    
        # Display Results
        st.markdown('# Max Likelihood Results')
        for par in self.maxlike_result['pbest'].values():
            st.text(repr(par))
        st.markdown('## Function calls: ' + str(self.maxlike_result['fcalls']))
        st.markdown('## ln(L): ' + str(-1 * self.maxlike_result['fbest']))
        st.markdown('## Red Chi 2: ' + str(self.optprob.scorer.compute_redchi2(self.maxlike_result['pbest'], include_white_error=True, include_kernel_error=True)))
    
        # Full RV plot
        st.markdown('## Full RV Plot')
        self.full_fig = self.optprob.plot_full_rvs(pars=self.maxlike_result["pbest"], n_model_pts=5000)
        self.comps["rvfigfull_maxlike"] = st.plotly_chart(self.full_fig)
    
        # Phased rv plot
        st.markdown('## Phased Planets')
        self.planet_figs = self.optprob.plot_phased_rvs_all(pars=self.maxlike_result["pbest"])
        for i, planet_index in enumerate(self.optprob.planets_dict):
            name = "figplanet_" + str(planet_index) + "_maxlike"
            self.comps[name] = st.plotly_chart(self.planet_figs[i])
            
        return self.comps
    
class ModelCompResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, mc_result):
        super().__init__(comps=comps, label="Optimize Result")
        self.optprob = optprob
        self.mc_result = mc_result
        self.write()
        
    def write(self):
        st.markdown('# Model Comparison')
        df = pd.DataFrame()
        n_models = len(self.mc_result)
        df["Planets"] = [""]*n_models
        df["ln \u2112"] = [1.0]*n_models
        df["\u0394 AICc"] = [1.0]*n_models
        df["\u0394 BIC"] = [1.0]*n_models
        df["N free"] = [1.0]*n_models
        df["Red. \u03C7 Sq."] = [1.0]*n_models
        for i in range(n_models):
            s = ""
            for planet_index in self.mc_result[i]["planets_dict"]:
                s += self.mc_result[i]["planets_dict"][planet_index]["label"] + ", "
            if len(s) > 0:
                s = s[0:-2]
            df["Planets"][i] = s
            df["ln \u2112"][i] = self.mc_result[i]["lnL"]
            df["\u0394 AICc"][i] = self.mc_result[i]["delta_aicc"]
            df["\u0394 BIC"][i] = self.mc_result[i]["delta_bic"]
            df["N free"][i] = self.mc_result[i]["pbest"].num_varied()
            df["Red. \u03C7 Sq."][i] = self.mc_result[i]["redchi2"]
        st.table(df)
        return self.comps
    
class MCMCResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, mcmc_result):
        super().__init__(comps=comps, label="MCMC Result")
        self.optprob = optprob
        self.mcmc_result = mcmc_result
        self.write()
        
    def write(self):
    
        # Display Results
        st.markdown('# MCMC Results')
        for par in self.mcmc_result['pmed'].values():
            st.text(repr(par))
        st.markdown('## ln(L): ' + str(self.mcmc_result['lnL']))
    
        # Full RV plot
        st.markdown('## Full RV Plot')
        self.full_fig = self.optprob.plot_full_rvs(pars=self.mcmc_result["pbest"], n_model_pts=5000)
        self.comps["rvfigfull_mcmc"] = st.plotly_chart(self.full_fig)
    
        # Phased rv plot
        st.markdown('## Phased Planets')
        self.planet_figs = self.optprob.plot_phased_rvs_all(pars=self.mcmc_result["pbest"])
        for i, planet_index in enumerate(self.optprob.planets_dict):
            name = "figplanet_" + str(planet_index) + "_mcmc"
            self.comps[name] = st.plotly_chart(self.planet_figs[i])
    
        # Corner plot
        st.markdown('## Corner Plot')
        self.corner_plot = self.optprob.corner_plot(mcmc_result=self.mcmc_result)
        self.comps["corner_plot"] = st.pyplot(self.corner_plot)
        
        return self.comps
        
class PlanetsResults(StreamlitComponent):
    
    def __init__(self, comps, optprob, sampler_result):
        super().__init__(comps=comps, label="Planets")
        self.optprob = optprob
        self.sampler_result = sampler_result
        self.write()
        
    def write(self):
    
        # Display Results
        st.markdown('# Planets')
        if len(self.optprob.planets_dict) > 0:
            planet_masses = self.optprob.compute_planet_masses(self.sampler_result)
            planet_rhos = self.optprob.compute_planet_densities(self.sampler_result)
            planet_smas = self.optprob.compute_semimajor_axis(self.sampler_result)
            for planet_index in self.optprob.planets_dict:
                
                # Mass
                s = self.optprob.star_name + " " + self.optprob.planets_dict[planet_index]["label"] + ": "
                s += str(planet_masses[planet_index][0])
                s += ", -" + str(planet_masses[planet_index][1])
                s += " +" + str(planet_masses[planet_index][2])
                s += " M Earth"
                st.text(s)
                
                # rho
                s = self.optprob.star_name + " " + self.optprob.planets_dict[planet_index]["label"] + ": "
                s += str(planet_rhos[planet_index][0])
                s += ", -" + str(planet_rhos[planet_index][1])
                s += " +" + str(planet_rhos[planet_index][2])
                s += " g/cm^3"
                st.text(s)
                
                # a
                s = self.optprob.star_name + " " + self.optprob.planets_dict[planet_index]["label"] + ": "
                s += str(planet_smas[planet_index][0])
                s += ", -" + str(planet_smas[planet_index][1])
                s += " +" + str(planet_smas[planet_index][2])
                s += " AU"
                st.text(s)
                
        else:
            st.text("None")
            
        return self.comps
      
class GLSResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, gls_result=None):
        super().__init__(comps=comps, label="Period Search Result")
        self.optprob = optprob
        self.gls_result = gls_result
        self.write()
        
    def write(self):
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        fig.add_trace(plotly.graph_objects.Scatter(x=1 / self.gls_result.f, y=self.gls_result.power, line=dict(color='black', width=1), showlegend=False), row=1, col=1)
        fig.update_xaxes(title_text='<b>Period [days]</b>', row=1, col=1)
        fig.update_yaxes(title_text='<b>Power</b>', row=1, col=1)
        fig.update_layout(template="ggplot2")
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        st.plotly_chart(fig)
        fig.write_html(self.optprob.output_path + self.optprob.star_name.replace(' ', '_') + '_glspgram_' + pcutils.gendatestr(time=True) + '.html')
        return self.comps
        
class RVPeriodSearchResult(StreamlitComponent):
        
    def __init__(self, comps, optprob, periods, persearch_result=None):
        super().__init__(comps=comps, label="Period Search Result")
        self.optprob = optprob
        self.periods = periods
        self.persearch_result = persearch_result
        self.write()
    
    def write(self):
        
        # Extract log L
        lnLs = np.array([-1 * result["fbest"] for result in self.persearch_result])
        
        # Create a plot and return results
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        fig.add_trace(plotly.graph_objects.Scatter(x=self.periods, y=lnLs, line=dict(color='black', width=1)), row=1, col=1)
        fig.update_xaxes(title_text='Period [days]', row=1, col=1)
        fig.update_yaxes(title_text='ln(L)', row=1, col=1)
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        st.plotly_chart(fig)
        fig.write_html(self.optprob.output_path + self.optprob.star_name.replace(' ', '_') + '_brute_force_pgram_' + pcutils.gendatestr(time=True) + '.html')
        return self.comps
        
class RVColorResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, rvcolor_result):
        super().__init__(comps=comps, label="RV Color Result")
        self.optprob = optprob
        self.rvcolor_result = rvcolor_result
        self.write()
        
    def write(self):
    
        # Display Results
        st.markdown('# RV Color Results')
        self.rvcolor_fig = self.optprob.plot_rvcolor(self.rvcolor_result)
        self.comps["rvcolor_fig"] = st.plotly_chart(self.rvcolor_fig)
        
        return self.comps