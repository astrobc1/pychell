import streamlit as st
import numpy as np
import importlib
import pychell.utils as pcutils

class RVOptimPage:
    
    def __init__(self):
        self.comps = {}
        self.run_user_file = False
        self.write()
        
    def write(self):
        st.title("RV Optimize")
        self.file_selector = UserFileSelector(comps=self.comps)
        if self.comps["load_script_button"] or self.run_user_file:
            self.run_user_file = True
            self.user_mod = pcutils.module_from_file(self.comps["user_script_input"])
            self.comps["user_mod"] = self.user_mod
            self.data_selector = DataSelector(comps=self.comps, data=self.user_mod.data)
            if len(self.data_selector.data) == 0:
                return None
            self.optprob = self.user_mod.optprob
            self.rv_actions = RVActions(self.comps, self.optprob)
            if self.comps["max_like_button"]:
                opt_result = self.optprob.optimize()
                self.maxlike_results = MaxLikeResult(self.comps, self.optprob, opt_result)
            if self.comps["sample_button"]:
                sampler_result = self.optprob.sample()
                self.sampler_results = MCMCResult(self.comps, self.optprob, sampler_result)
            if self.comps["per_search_button"]:
                psearch_result = self.optprob.rv_period_search(pars=None, pmin=self.comps["persearch_min_input"], pmax=self.comps["persearch_min_input"], n_periods=None, n_threads=1)
                self.sampler_results = PeriodSearchResult(self.comps, self.optprob, psearch_result)


class StreamlitComponent:
    
    def __init__(self, comps, label="", *args, **kwargs):
        self.comps = comps
        self.label = label
        
    def write(self, *args, **kwargs):
        pass
    
class UserFileSelector(StreamlitComponent):
    
    def __init__(self, comps):
        super().__init__(comps=comps, label="File Selector")
        self.write()
        
    def write(self):
        self.comps["user_script_input"] = st.text_input(label="Path", value="")
        self.comps["load_script_button"] = st.checkbox(label="Load Script")
        return self.comps

class DataSelector(StreamlitComponent):
    
    def __init__(self, comps, data):
        super().__init__(comps=comps, label="Data Selector")
        self.data = data
        self.write()
        
    def write(self):
        # Use or ignore any instruments as st checkboxes.
        for data in self.data.values():
            self.comps["use_" + data.label] = st.checkbox(label=data.label + " " + str(len(data.t)), value=True)
        self.update_data()
        return self.comps
    
    def update_data(self):
        for inst in list(self.data.keys()):
            if not self.comps["use_" + inst]:
                del self.data[inst]
    
class RVActions(StreamlitComponent):
    
    def __init__(self, comps, optprob):
        super().__init__(comps=comps, label="Data Selector")
        self.optprob = optprob
        self.write()
        
    def write(self):
        
        # Primary actions
        st.markdown('## Actions')
        self.comps["max_like_button"] = st.button(label='Max Like')
        self.comps["sample_button"] = st.button(label='MCMC')
        self.comps["model_comp_button"] = st.button(label='Model Comparison')
        self.comps["per_search_button"] = st.button('Period Search')
        
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
    
    def __init__(self, comps, optprob, opt_result):
        super().__init__(comps=comps, label="Optimize Result")
        self.optprob = optprob
        self.opt_result = opt_result
        self.write()
        
    def write(self):
    
        # Display Results
        st.markdown('# Optimize Results')
        st.text(repr(self.opt_result['pbest']))
        st.markdown('## Function calls: ' + str(self.opt_result['fcalls']))
        st.markdown('## ln(L): ' + str(-1 * self.opt_result['fbest']))
    
        # Full RV plot
        st.markdown('## Full RV Plot')
        plotly_fig = self.optprob.rv_plot(opt_result=self.opt_result, n_model_pts=5000)
        self.comps["rvfigfull_maxlike"] = st.plotly_chart(plotly_fig)
        plotly_fig.write_html(self.comps["user_mod"].path + self.optprob.star_name.replace(' ', '_') + '_rvs_full_maxlike' + pcutils.gendatestr(True) + '.html')
    
        # Phased rv plot
        st.markdown('## Phased Planets')
        for planet_index in self.optprob.planets_dict:
            name = "figplanet_" + str(planet_index) + "_maxlike"
            plotly_fig = self.optprob.rv_phase_plot(planet_index=planet_index, opt_result=self.opt_result)
            self.comps[name] = st.plotly_chart(plotly_fig)
            plotly_fig.write_html(self.comps["user_mod"].path + self.optprob.star_name.replace(' ', '_') + self.optprob.planets_dict[planet_index]["label"] + '_rvsphased_maxlike' + pcutils.gendatestr(True) + '.html')
            
        return self.comps
    
class ModelCompResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, mc_result):
        super().__init__(comps=comps, label="Optimize Result")
        self.optprob = optprob
        self.mc_result = mc_result
        self.write()
        
    def write(self):
        st.markdown('# Model Comparison')
        for i in range(len(self.mc_result)):
            self.comps["mc_result_" + str(i+1)] = st.write(self.mc_result[i])
        return self.comps
    
class MCMCResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, sampler_result):
        super().__init__(comps=comps, label="Optimize Result")
        self.optprob = optprob
        self.sampler_result = sampler_result
        self.write()
        
    def write(self):
    
        # Display Results
        st.markdown('# MCMC Results')
        st.text(repr(sampler_result['pbest']))
        st.text('Parameter Uncertainties: ')
        st.text(sampler_result['punc'])
        st.markdown('## ln(L): ' + str(sampler_result['lnL']))
    
        # Full RV plot
        st.markdown('## Full RV Plot')
        plotly_fig = self.optprob.rv_plot(opt_result=sampler_result, n_model_pts=5000)
        self.comps["rvfigfull_mcmc"] = st.plotly_chart(plotly_fig)
        plotly_fig.write_html(self.comps["user_mod"].path + self.optprob.star_name.replace(' ', '_') + '_rvs_full_mcmc_' + pcutils.gendatestr(True) + '.html')
    
        # Phased rv plot
        st.markdown('## Phased Planets')
        for planet_index in self.optprob.planets_dict:
            name = "figplanet_" + str(planet_index) + "_mcmc"
            plotly_fig = self.optprob.rv_phase_plot(planet_index=planet_index, opt_result=self.sampler_result)
            self.comps[name] = st.plotly_chart(plotly_fig)
            plotly_fig.write_html(self.comps["user_mod"].path + self.optprob.star_name.replace(' ', '_') + self.optprob.planets_dict[planet_index]["label"] + '_rvsphased_mcmc' + pcutils.gendatestr(True) + '.html')
    
        # Corner plot
        st.markdown('## Corner Plot')
        corner_plot = self.optprob.corner_plot(opt_result=self.sampler_result)
        self.comps["corner_plot"] = st.pyplot(corner_plot)
      
class PeriodSearchResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, psearch_result=None):
        super().__init__(comps=comps, label="Optimize Result")
        self.optprob = optprob
        self.write()
        
    def write(self):
        if comps["persearch_kind_input"] == "Brute Force":
            plotly_fig, period_search_results = self.optprob.rv_period_search(pmin=float(self.comps["period_min_input"]), pmax=float(self.comps["period_max_input"]), n_periods=1000, n_threads=8, planet_index=2)
            st.plotly_chart(fig)
            plotly_fig.write_html(self.comps["user_mod"].path + star_name.replace(' ', '_') + '_brute_force_pgram_' + pcutils.gendatestr(time=True) + '.html')
        else:
            plotly_fig = optprob.gls_periodogram(apply_gp=use_gp_input, remove_planets=self.comps["remove_planet_inputs"], pmin=float(self.comps["period_min_input"]), pmax=float(self.comps["period_max_input"]))
            st.plotly_chart(fig)
            plotly_fig.write_html(self.comps["user_mod"].path + self.optprob.star_name.replace(' ', '_') + '_glspgram_' + pcutils.gendatestr(time=True) + '.html')