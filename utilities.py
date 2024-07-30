import perceval as pcvl
from perceval.algorithm import Sampler
from perceval.components.unitary_components import PS, BS
import numpy as np

class VariationalModule():
    """
    VariationalModule class
    """
    def __init__(
            self,
            circuit,
            initial_params,
            input_state,
            postselect_func,
            remote=False,
            remote_config={
                'processor': 'sim:ascella',
                'qpu_params':{
                    'HOM': .9244,
                    'transmittance': .0565,
                    'g2': .0053
                },
                'nshots': 1e5
            }
                 ):
        
        self.circuit = circuit
        self.in_state = input_state
        self.postselect_func = postselect_func
        for i, p in enumerate(self.circuit.get_parameters()):
            p.set_value(initial_params[i])

        self.set_remote(remote, remote_config)

    def compute_pdf(self, parameter_iteration=False, n_iter=None):
        if self.remote:
            nsamples = self.nshots
            if parameter_iteration:
                self.sampler.default_job_name = f"{self.nshots} sampling with {n_iter} parameter iterations"
            else: 
                self.sampler.default_job_name = f"{self.nshots} sampling"
            remote_job = self.sampler.sample_count.execute_async(nsamples)
            while not remote_job.is_complete:
                continue
            results = remote_job.get_results()
            if not parameter_iteration:
                results = [results]
            else:
                results = results['results_list']
                probs_list = []
            for result in results:
                result['results'] = {k: v for k, v in result['results'].items() if self.postselect_func(k)}
                tot_samples = np.sum([v for v in result['results'].values()])
                if tot_samples != nsamples:
                    tot_probs = [v/tot_samples for v in result['results'].values()]
                    sampled_indeces = np.random.choice(range(len(result['results'])), nsamples, p=tot_probs)
                    new_results = {k: 0 for i, k in enumerate(result['results'].keys()) if i in sampled_indeces}
                    for i in sampled_indeces:
                        new_results[list(result['results'].keys())[i]] += 1
                    result['results'] = new_results
                
                probs = {k: v/nsamples for k, v in result['results'].items()}

                if not parameter_iteration:
                    return probs
                else:
                    probs_list.append(probs)
            return probs_list
        else:
            probs = self.sampler.probs()["results"]
        return probs

    def sample(self, n_samples=1):
        sampled_indeces = np.random.choice(range(len(self.probs.keys())), n_samples, p=list(self.probs.values()))
        results = {k: 0 for i, k in enumerate(self.probs.keys()) if i in sampled_indeces}
        for i in sampled_indeces:
            results[list(self.probs.keys())[i]] += 1
        return results
    
    def pdf(self):
        return self.probs
    
    def launch_iterator(self, parameter_list):
        # format the parameter list
        parameter_list = [{p.name: parameter_sublist[i] for i, p in enumerate(self.circuit.get_parameters())} for parameter_sublist in parameter_list]
        # store the initial parameters
        params_bkp = [p._value for p in self.circuit.get_parameters()]
        # reset current parameters
        for p in self.circuit.get_parameters():
            p.reset()
        # update circuit
        self.processor.set_circuit(self.circuit)

        # add the parameter iterator to the processor
        self.processor.set_parameter("parameter_iterator", parameter_list)

        # sample
        self.sampler = pcvl.algorithm.Sampler(self.processor)
        probs_list = self.compute_pdf(parameter_iteration=True, n_iter=len(parameter_list))

        # restore the initial parameters
        for i, p in enumerate(self.circuit.get_parameters()):
            p.set_value(params_bkp[i])
        # update circuit
        self.processor.set_circuit(self.circuit)
        # remove the parameter iterator from the processor
        self.processor._parameters.pop("parameter_iterator")
        
        return probs_list

    
    def updateCircuit(self, new_params):
        for i, p in enumerate(self.circuit.get_parameters()):
            p.set_value(new_params[i])
        # update the processor
        if self.remote:
            self.processor.set_circuit(self.circuit)
        else:
            self.processor = pcvl.Processor("SLOS", self.circuit)
            self.processor.with_input(self.in_state)
            self.processor.set_postprocess(self.postselect_func)

        self.sampler = pcvl.algorithm.Sampler(self.processor)
        self.probs = self.compute_pdf() 

    def set_remote(self, flag=True, remote_config=None):
        self.remote = flag

        if self.remote:
            # configure the remote params
            self.processor = pcvl.RemoteProcessor(remote_config['processor'], remote_config['token'])

            self.processor.set_circuit(self.circuit)
            self.processor.with_input(self.in_state)
            self.processor.set_parameters({
                'HOM': remote_config['qpu_params']['HOM'],
                'transmittance': remote_config['qpu_params']['transmittance'],
                'g2': remote_config['qpu_params']['g2']
            })
            self.processor.set_parameter('mode_post_select', 1)
            self.nshots = remote_config['nshots']
        else:
            # processor
            self.processor = pcvl.Processor("SLOS", self.circuit)
            self.processor.with_input(self.in_state)
            self.processor.set_postprocess(self.postselect_func)

        # sampler 
        self.sampler = pcvl.algorithm.Sampler(self.processor)
        self.probs = self.compute_pdf()

# define the class tree as a child class of VariationalModule with a pre-defined circuit
class Tree(VariationalModule):
    def __init__(
        self, 
        n_leaves=4,
        remote=False,
        remote_config={
            'processor': 'sim:ascella',
            'qpu_params':{
                'HOM': .9244,
                'transmittance': .0565,
                'g2': .0053
            },
            'nshots': 1e5
        }
                ):
        self.n_leaves = n_leaves
        # create the circuit
        circuit = self.build_circuit()
        # initialize the parameters
        initial_params = [np.pi/4]*len(circuit.get_parameters())
        # input state
        in_list = [0]*(self.n_leaves+self.bias)
        in_list[self.layers-1] = 1
        in_state = pcvl.BasicState(in_list)

        # initialize the parent class
        VariationalModule.__init__(self, circuit, initial_params, in_state, self.postselect_func, remote, remote_config)


    def build_circuit(self):
        # put bias if leaves are odd
        self.bias = 1 if self.n_leaves%2 else 0

        # create the tree
        layers = int((self.bias+self.n_leaves)/2)
        self.layers = layers
        mzi_indeces = [[[0]*2]*(i+1) for i in range(layers)]
        i_0 = layers
        for s in range(layers):
            if s==0:
                mzi_indeces[s][0] = [i_0, i_0-1]
            else:
                z = 0
                for i, j in mzi_indeces[s-1]:
                    if [i+1, i] not in mzi_indeces[s]:
                        mzi_indeces[s][z] = [i+1, i]
                        z += 1
                    if [j, j-1] not in mzi_indeces[s]:
                        mzi_indeces[s][z] = [j, j-1]
                        z += 1

        circuit = pcvl.Circuit(self.n_leaves+self.bias)
        mzi_counter = 0
        for s in range(self.layers):
            for mzi in mzi_indeces[s]:
                circuit.add(int(mzi[1]), BS())
                circuit.add(int(mzi[1]), PS(pcvl.P(f"phi_{mzi_counter}")))
                mzi_counter += 1
                circuit.add(int(mzi[1]), BS())
                circuit.add(int(mzi[1]), PS(pcvl.P(f"phi_{mzi_counter}")))
                mzi_counter += 1
                if s==self.layers-1:
                    circuit.add(int(mzi[0]), PS(pcvl.P(f"phi_{mzi_counter}")))
                    mzi_counter += 1
        
        return circuit
    
    def postselect_func(self, s: pcvl.BasicState) -> bool:
        if self.bias:
            if s[self.n_leaves] == 1:
                return False
        counts = np.sum([n for n in s])
        if counts !=1:
            return False
        return True
    
    # env friendly state to mode function
    def stateToModeIdx(self, state):
        state_str = str(state)
        state_str = state_str.replace("|", "")
        state_str = state_str.replace(">", "")
        state_str = state_str.split(",")
        return state_str.index("1")
    
    # costumize sample and pdf functions
    def sample(self):
        results = VariationalModule.sample(self, n_samples=1)
        mode = self.stateToModeIdx([k for k in results.keys()][0])
        return mode
    
    def pdf(self):
        results = VariationalModule.pdf(self)
        results = {self.stateToModeIdx(k):v for k,v in results.items()}
        return results
    
class FullMesh():
    def __init__(
            self, 
            n_leaves=4, 
            remote=False, 
            remote_config={
                'processor': 'sim:ascella',
                'qpu_params':{
                    'HOM': .9244,
                    'transmittance': .0565,
                    'g2': .0053
                },
                'nshots': 1e5
            }
                 ):
        
        self.n_leaves = n_leaves
        # put bias if leaves are odd
        self.bias = 1 if n_leaves%2 else 0

        # create the tree
        layers = int((self.bias+self.n_leaves)/2)
        self.layers = layers
        mzi_indeces = [[[0]*2]*(i+1) for i in range(layers)]
        i_0 = layers
        for s in range(layers):
            if s==0:
                mzi_indeces[s][0] = [i_0, i_0-1]
            else:
                z = 0
                for i, j in mzi_indeces[s-1]:
                    if [i+1, i] not in mzi_indeces[s]:
                        mzi_indeces[s][z] = [i+1, i]
                        z += 1
                    if [j, j-1] not in mzi_indeces[s]:
                        mzi_indeces[s][z] = [j, j-1]
                        z += 1
                        
        self.circuit = pcvl.Circuit(self.n_leaves+self.bias)
        mzi_counter = 0
        for s in range(self.layers):
            for mzi in mzi_indeces[s]:
                self.circuit.add(int(mzi[1]), BS())
                self.circuit.add(int(mzi[1]), PS(pcvl.P(f"phi_t_{mzi_counter}")))
                mzi_counter += 1
                self.circuit.add(int(mzi[1]), BS())
                self.circuit.add(int(mzi[1]), PS(pcvl.P(f"phi_t_{mzi_counter}")))
                mzi_counter += 1
                if s==self.layers-1:
                    self.circuit.add(int(mzi[0]), PS(pcvl.P(f"phi_t_{mzi_counter}")))
                    mzi_counter += 1

        # concatenate the square mesh
        self.mzi_counter = 0
        for i in range(self.n_leaves):
            if i%2 == 0:
                for j in range(0, self.n_leaves, 2):
                    if j == self.n_leaves-1: continue
                    self.circuit.add(int(j), BS())
                    self.circuit.add(int(j), PS(pcvl.P(f"phi_c_{self.mzi_counter}")))
                    self.mzi_counter += 1
                    self.circuit.add(int(j), BS())
                    if i != self.n_leaves-1:
                        self.circuit.add(int(j), PS(pcvl.P(f"phi_c_{self.mzi_counter}")))
                        self.mzi_counter += 1
            else:
                for j in range(1, self.n_leaves, 2):
                    if j == self.n_leaves-1: continue
                    self.circuit.add(int(j), BS())
                    self.circuit.add(int(j), PS(pcvl.P(f"phi_c_{self.mzi_counter}")))
                    self.mzi_counter += 1
                    self.circuit.add(int(j), BS())
                    if i != self.n_leaves-1:
                        self.circuit.add(int(j), PS(pcvl.P(f"phi_c_{self.mzi_counter}")))
                        self.mzi_counter += 1
            # add last column of phase shifters
            if i == self.n_leaves-1:
                for j in range(0, self.n_leaves):
                    if j ==0: continue
                    self.circuit.add(int(j), PS(pcvl.P(f"phi_c_{self.mzi_counter}")))
                    self.mzi_counter += 1

        # initialize parameters
        for param in self.circuit.get_parameters():
            param._value = np.pi/4

        # input state
        in_list = [0]*(self.n_leaves+self.bias)
        in_list[self.layers-1] = 1
        self.in_state = pcvl.BasicState(in_list)

        # flag for remote computation
        self.remote = remote
        if self.remote:
            # configure the remote params
            self.processor = pcvl.RemoteProcessor(remote_config['processor'], remote_config['token'])

            self.processor.set_circuit(self.circuit)
            self.processor.with_input(self.in_state)
            self.processor.set_parameters({
                'HOM': remote_config['qpu_params']['HOM'],
                'transmittance': remote_config['qpu_params']['transmittance'],
                'g2': remote_config['qpu_params']['g2']
            })
            self.processor.set_parameter('mode_post_select', 1)
            self.nshots = remote_config['nshots']
        else:
            # processor
            self.processor = pcvl.Processor("SLOS", self.circuit)
            self.processor.with_input(self.in_state)
            self.processor.set_postprocess(self.postselect_func)

        # sampler 
        self.sampler = pcvl.algorithm.Sampler(self.processor)
        self.probs = self.compute_pdf()

    # post_select on the first two modes
    def postselect_func(self, s: pcvl.BasicState) -> bool:
        counts = np.sum([n for n in s])
        if counts ==1:
            if s[0] + s[1] > 0:
                return True
        return False

    # take perceval state and return the index of the mode with 1
    def stateToModeIdx(self, state):
        state_str = str(state)
        state_str = state_str.replace("|", "")
        state_str = state_str.replace(">", "")
        state_str = state_str.split(",")
        return state_str.index("1")
    
    # perform single photon walk
    def sample(self):
        """"
        nsamples = 1
        if self.remote:
            self.sampler.default_job_name = "single_photon_walk"
            remote_job = self.sampler.sample_count.execute_async(nsamples)
            while not remote_job.is_complete:
                continue
            results = remote_job.get_results()
            if len(result['results']) == 0:
                while len(result['results']) == 0:
                    remote_job = self.sampler.sample_count.execute_async(nsamples)
                    while not remote_job.is_complete:
                        continue
                    results = remote_job.get_results()
            # postselect
            result['results'] = {k: v for k, v in result['results'].items() if self.postselect_func(k)}
            tot_samples = np.sum([v for v in result['results'].values()])
            if tot_samples != nsamples:
                tot_probs = [v/tot_samples for v in result['results'].values()]
                sampled_indeces = np.random.choice(range(len(result['results'])), nsamples, p=tot_probs)
                new_results = {k: 0 for i, k in enumerate(result['results'].keys()) if i in sampled_indeces}
                for i in sampled_indeces:
                    new_results[list(result['results'].keys())[i]] += 1
                result['results'] = new_results
        else:
            results = self.sampler.sample_count(nsamples)
            while len(result['results']) == 0:
                results = self.sampler.sample_count(nsamples)
        # return the mode index
        return self.stateToModeIdx(list(results["results"].keys())[0])
        """
        return np.random.choice(list(self.probs.keys()), p=list(self.probs.values()))

    
    def compute_pdf(self, parameter_iteration=False, n_iter=None):
        if self.remote:
            nsamples = self.nshots
            if parameter_iteration:
                self.sampler.default_job_name = f"{self.nshots} sampling with {n_iter} parameter iterations"
            else: 
                self.sampler.default_job_name = f"{self.nshots} sampling"
            remote_job = self.sampler.sample_count.execute_async(nsamples)
            while not remote_job.is_complete:
                continue
            results = remote_job.get_results()
            if not parameter_iteration:
                results = [results]
            else:
                results = results['results_list']
                probs_list = []
            for result in results:
                tot_samples = np.sum([v for v in result['results'].values()])
                if tot_samples != nsamples:
                    tot_probs = [v/tot_samples for v in result['results'].values()]
                    sampled_indeces = np.random.choice(range(len(result['results'])), nsamples, p=tot_probs)
                    new_results = {k: 0 for i, k in enumerate(result['results'].keys()) if i in sampled_indeces}
                    for i in sampled_indeces:
                        new_results[list(result['results'].keys())[i]] += 1
                    result['results'] = new_results
            
                # postselect
                result['results'] = {k: v for k, v in result['results'].items() if self.postselect_func(k)}
                tot_samples = np.sum([v for v in result['results'].values()])
                probs = {k: v/tot_samples for k, v in result['results'].items()}
                probs = {self.stateToModeIdx(state): prob for state, prob in probs.items()}
                
                if not parameter_iteration:
                    return probs
                else:
                    probs_list.append(probs)
            return probs_list
        else:
            probs = self.sampler.probs()["results"]
        probs = {self.stateToModeIdx(state): prob for state, prob in probs.items()}
        return probs
    
    def pdf(self):
        return self.probs
    
    def launch_iterator(self, parameter_list):
        # store the initial parameters
        params_bkp = [p._value for p in self.circuit.get_parameters()]
        # reset current parameters
        for p in self.circuit.get_parameters():
            p.reset()
        # update circuit
        self.processor.set_circuit(self.circuit)

        # add the parameter iterator to the processor
        self.processor.set_parameter("parameter_iterator", parameter_list)

        # sample
        self.sampler = pcvl.algorithm.Sampler(self.processor)
        probs_list = self.compute_pdf(parameter_iteration=True, n_iter=len(parameter_list))

        # restore the initial parameters
        for i, p in enumerate(self.circuit.get_parameters()):
            #print(params_bkp[i])
            p.set_value(params_bkp[i])
        # update circuit
        self.processor.set_circuit(self.circuit)
        # remove the parameter iterator from the processor
        self.processor._parameters.pop("parameter_iterator")
        
        return probs_list
        
    def updateCircuit(self, new_params):
        for i, p in enumerate(self.circuit.get_parameters()):
            p.set_value(new_params[i])
        # update the processor
        if self.remote:
            self.processor.set_circuit(self.circuit)
        else:
            self.processor = pcvl.Processor("SLOS", self.circuit)
            self.processor.with_input(self.in_state)
            self.processor.set_postprocess(self.postselect_func)

        self.sampler = pcvl.algorithm.Sampler(self.processor)
        self.probs = self.compute_pdf()

    def updateTree(self, new_params):
        i = 0
        for p in self.circuit.get_parameters():
            if "t" in p.name:
                p.set_value(new_params[i])
                i += 1
        # update the processor
        if self.remote:
            self.processor.set_circuit(self.circuit)
        else:
            self.processor = pcvl.Processor("SLOS", self.circuit)
            self.processor.with_input(self.in_state)
            self.processor.set_postprocess(self.postselect_func)

        self.sampler = pcvl.algorithm.Sampler(self.processor)
        self.probs = self.compute_pdf()
    
    def updateMesh(self, new_params):
        i = 0
        for p in self.circuit.get_parameters():
            if "c" in p.name:
                p.set_value(new_params[i])
                i += 1
        # update the processor
        if self.remote:
            self.processor.set_circuit(self.circuit)
        else:
            self.processor = pcvl.Processor("SLOS", self.circuit)
            self.processor.with_input(self.in_state)
            self.processor.set_postprocess(self.postselect_func)

        self.sampler = pcvl.algorithm.Sampler(self.processor)
        self.probs = self.compute_pdf()

# create class Percept
# the Percept has n_observables hidden variables, each variable has n_values possible values
# and whenever the Percept is initialized the hidden variables are fixed
# the observables of the Percept are stored in a dictionary, where the key is the name of the observable
# and the value is the value of the observable
class Percept():
    """
    Percept class
    """
    def __init__(self, observables_dict):
        # initialize internal variables as self.key = value
        for key, value in observables_dict.items():
            setattr(self, key, value)
    
    # return the value of the Percept given the name of the observable
    def get_value(self, param_name):
        return getattr(self, param_name)
    
    # set the value of the Percept given the name of the observable
    def set_value(self, param_name, value):
        setattr(self, param_name, value)