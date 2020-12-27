import streamlit as st
import numpy as np
import importlib

class RVOptimPage:
    
    def __init__(self):
        self.comps = {}
        self.write()
        
    def write(self):
        st.title("RV Optimize")
        self.file_selector = UserFileSelector(comps=self.comps)
        if self.comps["load_script_button"]:
            self.user_mod = importlib.import_module(self.user_sc)
            self.optprob = self.user_mod.optprob
            self.data_selector = DataSelector(self.comps, self.optprob)
            self.rv_actions = RVActions(self.comps, self.optprob)
            if self.comps["max_like_button"]:
                self.maxlike_results = OptimizeResult(self.comps, self.optprob)
            if self.comps["sample_button"]:
                self.sampler_results = MCMCResult(self.comps, self.optprob)
            if self.comps["per_search_button"]:
                self.sampler_results = PeriodSearchResult(self.comps, self.optprob)


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
        self.comps["load_script_button"] = st.button(label="Load Script")
        return self.comps

class DataSelector(StreamlitComponent):
    
    def __init__(self, comps, data):
        super().__init__(label="data_selector")
        self.data = data
        self.write()
        
    def write(self):
        # Use or ignore any instruments as st checkboxes.
        for data in self.data.values():
            self.comps["use_" + data.label] = st.checkbox(label=data.label + " " + str(len(data.t)), value=True)
        self.update_data()
        return self.comps
    
    def update_data(self):
        for data in self.data.values():
            if not self.comps["use_" + data.label]:
                del self.data[data.label]
    
class RVActions(StreamlitComponent):
    
    def __init__(self, comps, optprob):
        super().__init__(label="Data Selector")
        self.optprob = optprob
        self.write()
        
    def write(self):
        
        # Primary actions
        st.markdown('## Actions')
        self.comps["max_like_button"] = st.button(label='Run Max Like')
        self.comps["sample_button"] = st.button(label='Run MCMC')
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
        
        self.comps = {}
    
        # Display Results
        st.markdown('# Optimize Results')
        st.text(repr(opt_result['pbest']))
        st.markdown('## Function calls: ' + str(self.opt_result['fcalls']))
        st.markdown('## ln(L): ' + str(-1 * self.opt_result['fbest']))
    
        # Full RV plot
        st.markdown('## Full RV Plot')
        self.comps["figfull"] = optprob.rv_plot(opt_result=opt_result, n_model_pts=5000)
        st.plotly_chart(figfull)
        self.comps["figfull"].write_html(self.optprob.star_name.replace(' ', '_') + '_rvs_full_' + pcutils.gendatestr(True) + '.html')
    
        # Phased rv plot
        st.markdown('## Phased Planets')
        for planet_index in self.optprob.planets_dict:
            name = "figplanet_" + str(planet_index)
            plotly_fig = optprob.rv_phase_plot(planet_index=planet_index, opt_result=self.opt_result)
            self.comps[name] = st.plotly_chart(plotly_fig)
            
        return self.comps
    
class ModelCompResult(StreamlitComponent):
    
    def __init__(self, comps=comps, optprob, mc_result):
        super().__init__(label="Optimize Result")
        self.optprob = optprob
        self.mc_result = mc_result
        self.write()
        
    def write(self):
        self.comps = {}
        st.markdown('# Model Comparison')
        for i in range(len(self.mc_result)):
            self.comps["mc_result_" + str(i+1)] = st.write(self.mc_result[i])
        return self.comps
    
class MCMCResult(StreamlitComponent):
    
    def __init__(self, optprob, sampler_result):
        super().__init__(comps=comps, label="Optimize Result")
        self.optprob = optprob
        self.sampler_result = sampler_result
        self.write()
        
    def write(self):
        self.comps = {}
    
        # Display Results
        st.markdown('# MCMC Results')
        st.text(repr(sampler_result['pbest']))
        st.text('Parameter Uncertainties: ')
        st.text(sampler_result['punc'])
        st.markdown('## ln(L): ' + str(sampler_result['lnL']))
    
        # Full RV plot
        st.markdown('## Full RV Plot')
        self.comps["figfull"] = optprob.rv_plot(opt_result=sampler_result, n_model_pts=5000)
        st.plotly_chart(self.comps["figfull"])
    
        # Phased rv plot
        st.markdown('## Phased Planets')
        for planet_index in self.optprob.planets_dict:
            name = "figplanet_" + str(planet_index)
            plotly_fig = optprob.rv_phase_plot(planet_index=planet_index, opt_result=self.sampler_result)
            self.comps[name] = st.plotly_chart(plotly_fig)
    
        # Corner plot
        st.markdown('## Corner Plot')
        corner_plot = optprob.corner_plot(opt_result=self.sampler_result)
        self.comps["corner_plot"] = st.pyplot(corner_plot)
      
class PeriodSearchResult(StreamlitComponent):
    
    def __init__(self, comps, optprob, psearch_result=None):
        super().__init__(comps=comps, label="Optimize Result")
        self.optprob = optprob
        self.write()
        
    def write(self):
        if comps["persearch_kind_input"] == "Brute Force":
            fig, period_search_results = optprob.rv_period_search(pmin=float(comps["period_min_input"]), pmax=float(comps["period_max_input"]), n_periods=1000, n_threads=8, planet_index=2)
            st.plotly_chart(fig)
            fig.write_html(star_name.replace(' ', '_') + '_brute_force_pgram_' + pcutils.gendatestr(time=True) + '.html')
        else:
            fig = optprob.gls_periodogram(apply_gp=use_gp_input, remove_planets=comps["remove_planet_inputs"], pmin=float(comps["period_min_input"]), pmax=float(comps["period_max_input"]))
            st.plotly_chart(fig)
            fig.write_html(star_name.replace(' ', '_') + '_glspgram_' + pcutils.gendatestr(time=True) + '.html')