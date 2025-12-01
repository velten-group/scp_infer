import numpy as np
import pandas as pd
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy as sc
import sergio_kd.sergio as sergio
from tqdm import tqdm


import scp_infer as scpi


class HiddenPrints:
    """Suppress print statements in a block of code."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def empty_dir(dir_path: str):
    """Empty a directory.

    Args:
        dir_path: path to the directory to empty
    """
    # empty the output directory
    outp_files = os.listdir(dir_path)
    if len(outp_files) > 0:
        print('removing previous files in: ', dir_path)
        for f in outp_files:
            os.remove(os.path.join(dir_path, f))


def write_data(data: list, file_path: str):
    """Write data to a file.

    Args:
        data: data to write to the file, list of lists
        file_path: path to the file to write to
    """
    f = open(file_path, "w")
    for row in data:
        for i, elem in enumerate(row):
            f.write(str(elem))

            if i < len(row) - 1:
                f.write(", ")
        f.write("\n")
    f.close()


def read_data(file_path: str):
    """Read data from a file.

    Args:
        file_path: path to the file to read from

    Returns:
        list: data read from the file
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(line.strip().split(", "))
    return data


def erdos_renyi(n_genes: int, p: float):
    """Generate a random Erdős–Rényi graph with n_genes genes and edge probability p.

    grph should be edited to be a DAG

    Args:
        n_genes: number of genes in the graph
        p: edge probability

    Returns:
        np.ndarray: adjacency matrix of the generated graph
    """
    base_graph = np.random.binomial(1, p, (n_genes, n_genes))
    # make sure the graph is a DAG
    for i in range(n_genes):
        for j in range(n_genes):
            if i == j:
                base_graph[i][j] = 0
            if base_graph[i][j] == 1 and base_graph[j][i] == 1:
                base_graph[i][j] = 0
    return base_graph


def plot_dag(graph: nx.Graph, n_plots: int):
    """Plot a directed acyclic graph with topological ordering using networkx.

    Args:
        graph: networkx graph


    Returns:
        None
    """
    G = graph
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    pert_nodes = [node for node in G if G.pred[node] != {}]

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal')
    for k in pos:
        pos[k][-1] *= -1

    fig, ax = plt.subplots(1, n_plots+1, figsize=(10*((n_plots+1)/2), 5))
    # 1. draw base Graph
    # color source nodes red
    color_map = []
    for node in G:
        if G.pred[node] == {}:
            color_map.append('green')
        else:
            color_map.append('blue')

    nx.draw_networkx(G, pos=pos, ax=ax[0], node_color=color_map)
    ax[0].set_title("Base Graph")

    # 2. draw Graph with perturbation
    for i, pert_node in enumerate(pert_nodes[:n_plots]):
        pertgraph = G.copy()
        for parent in list(pertgraph.pred[pert_node].keys()):
            pertgraph.remove_edge(parent, pert_node)
        color_map_p = color_map.copy()
        color_map_p[pert_node] = 'red'
        nx.draw_networkx(pertgraph, pos=pos,
                         ax=ax[i+1], node_color=color_map_p)
        ax[i+1].set_title(f"Perturbed Graph {i}")

    fig.tight_layout()
    plt.show()


def generate_GRN_file(graph: np.ndarray, file_path: str, hill_coefficient: float = 2, hill_K_low: float = 1, hill_K_high: float = 5):
    """
    Generate a base GRN with n_genes genes.

    also specify weigth and hill-coefficient for each edge. e.g. k ~ U(1,5), hc = 2

    file format:
    columns: target, num_regs, reg1, ... regN, weight1, ... weightN, hill_coeff1, ... hill_coeffN
    rows: target1, ... targetN

    Args:
        graph: adjacency matrix of the generated graph
        file_path: path to the file to write the graph to
        hill_coefficient: hill coefficient for the edges
        hill_K_low: lower limit for the hill coefficient
        hill_K_high: upper limit for the hill coefficient

    Returns:
        None
    """
    targets = []
    num_regs = []
    regs = []
    weights = []
    hill_coeffs = []

    # read the graph
    for i_target, row in enumerate(graph):
        regs_i = np.where(row == 1)[0]
        num_regs_i = len(regs_i)
        if num_regs_i == 0:
            continue
        targets.append(i_target)
        num_regs.append(num_regs_i)
        regs.append(regs_i)
        weights.append(np.random.uniform(hill_K_low, hill_K_high, num_regs_i))
        hill_coeffs.append(np.ones(num_regs_i) * hill_coefficient)

    # get into tabular format
    table = []
    for i in range(len(targets)):
        row = [targets[i], num_regs[i]]
        row.extend(regs[i])
        row.extend(weights[i])
        row.extend(hill_coeffs[i])
        table.append(row)
    write_data(table, file_path)


def generate_MR_expression_file(n_genes: int, GRN_file: str, file_path: str, n_cell_types: int = 1, MR_low: float = 3, MR_high: float = 4):
    """
    Generate master regulator file for completed GRN.

    Sample the expression levels for each regulator. 
    single cell type: e.g. sample from uniform distribution
    multiple cell types: e.g. switch between low and high epxression ranges (not iplemented)

    file format:
    columns: master_regulator, base_rate1, ... base_rateN
    rows: m_reg1, ... m_regN

    Args:
        GRN_file: path to the GRN file
        file_path: path to the file to write the master regulator file to
        n_cell_types: number of cell types
        MR_low: low expression value for the master regulator
        MR_high: high expression value for the master regulator
    """
    GRN = read_data(GRN_file)
    target_genes = [int(row[0]) for row in GRN]
    print('target genes:', target_genes)
    master_regulators = [i for i in range(n_genes) if i not in target_genes]
    print('master regulators:', master_regulators)
    table = []
    for i in master_regulators:
        row = [i]
        # add n_cell_types base rates
        row.extend(np.random.uniform(MR_low, MR_high, n_cell_types))
        table.append(row)
    write_data(table, file_path)


def generate_perturbation_files(n_genes, base_GRN_file, base_MR_file, output_dir, mean_exprs, KD_percent=0.1):
    """
    Generate perturbations on GRN.

    knockout each gene once and create a new GRN and MR file for each perturbation.

    Args:
        n_genes: number of genes in the GRN
        base_GRN_file: path to the base GRN file
        base_MR_file: path to the base MR file
        data_dir: directory to write the perturbed GRN and MR files to
        KD_low: low expression value for the perturbed gene
        KD_high: high expression value for the perturbed gene
    """
    empty_dir(output_dir)
    base_MR = read_data(base_MR_file)
    n_cell_types = len(base_MR[0]) - 1
    for perturbed_gene in range(n_genes):
        base_GRN = read_data(base_GRN_file)
        base_MR = read_data(base_MR_file)
        # 1. remove incoming edges to gene i if it is in GRN
        GRN = []
        for row in base_GRN:
            target = int(row[0])
            if target == perturbed_gene:
                continue
            else:
                GRN.append(row)

        # 2. replace gene i in MR with low expression values
        perturbed_row = [perturbed_gene]
        perturbed_row.extend(
            [KD_percent * mean_exprs[perturbed_gene]] * n_cell_types)

        master_regulators = [int(row[0]) for row in base_MR]
        # print('master regulators:', master_regulators)
        print('perturbed gene:', perturbed_gene)
        if perturbed_gene in master_regulators:
            print('replacing row', master_regulators.index(perturbed_gene))
            base_MR[master_regulators.index(perturbed_gene)] = perturbed_row
        else:
            base_MR.append(perturbed_row)

        # verification check: num_targets + num_masters = n_genes
        n_sum = len(GRN) + len(base_MR)
        assert n_sum == n_genes, f"n_sum = {n_sum}, n_genes = {n_genes}"

        # write the perturbed GRN and MR to files
        perturbed_GRN_file = os.path.join(
            output_dir, f"gene_{perturbed_gene}_perturbed_GRN.csv")
        perturbed_MR_file = os.path.join(
            output_dir, f"gene_{perturbed_gene}_perturbed_MR.csv")
        write_data(GRN, perturbed_GRN_file)
        write_data(base_MR, perturbed_MR_file)


def run_simulation(
        GRN_folders: list,
        output_dir: str,
        number_genes: int = 100,
        number_bins: int = 9,
        number_sc: int = 300,
        noise_params: int = 1,
        noise_type: str = 'dpd',
        decays: float = 0.8,
        sampling_state: int = 15,
        dt: float = 0.1,
        number_replicates: int = 1,
        hill_coefficient: float = 2,
        half_responses: list | None = None,
        verbose: bool = False
):
    """
    Run the simulation for each perturbation.
    Saves the expression data to files.

    Args:
        GRN_folders: list of paths to the GRN and MR files
        output_dir: directory to write the output files to
        number_genes: number of genes in the GRN
        number_bins: number of bins
        number_sc: number of single cells
        noise_params: noise parameters
        noise_type: type of noise
        decays: decay rate
        sampling_state: sampling state
        dt: time step
        number_replicates: number of replicates, i.e. repeated runs of the simulation
        hill_coefficient: hill coefficient for the edges
        half_responses: half responses hill functions
    """
    sim = sergio(number_genes, number_bins, number_sc, noise_params,
                 noise_type, decays, sampling_state=sampling_state, half_responses=half_responses, dt=dt)
    out_dir = output_dir

    # load base and perturbed GRN and MR files
    MR_files = []
    GRN_files = []
    for input_dir in GRN_folders:
        files = os.listdir(input_dir)
        files.sort()
        MR_files.extend([os.path.join(input_dir, f)
                        for f in files if 'MR' in f])
        GRN_files.extend([os.path.join(input_dir, f)
                         for f in files if 'GRN' in f])

    for i, MR_file, GRN_file in zip(range(len(MR_files)), MR_files, GRN_files):
        # determine label for output file
        perturbed = False
        if ('perturbed' in MR_file):
            perturbed = True
            perturbed_gene = int(MR_file.split('_')[-3])
            if verbose:
                print("perturbed gene: ", perturbed_gene)
            label = f'gene_{perturbed_gene}_perturbed'
        else:
            label = 'non-targeting'

        print("label: ", label)
        if verbose:
            print(MR_file, GRN_file)
            print("building graph")
            print("running perturbed simulation: ", perturbed)

        for replicate in tqdm(range(number_replicates)):
            # build graph and simulate
            sim.build_graph(
                GRN_file,
                MR_file,
                shared_coop_state=hill_coefficient
            )
            with HiddenPrints():
                sim.simulate()
            expr = sim.getExpressions()
            filename = f'{label}_replicate{replicate}_expr.npy'
            np.save(os.path.join(out_dir, filename), expr)


def get_mean_expressions(raw_expr_file: str):
    """
    Get the mean expression of the genes from a unedited expression file. (3D-numpy array)

    Args:
        raw_expr_file: path to the raw expression file

    Returns:
        list: mean expression of the genes
    """
    expr = np.load(raw_expr_file)
    expr = np.concatenate(expr, axis=1)
    mean_expr = np.mean(expr, axis=1)
    return mean_expr


def merge_outputs():
    raise NotImplementedError


def sample_dropout(dataset, dropout_k, percentile, rate_per_gene=False):
    """
    Sample dropouts for raw simulated data using a sigmoid dropout probability.

    Args:
        dataset: raw data
        percentile: percentile of the dropout distribution

    Returns:
        tuple: tuple containing:
            np.ndarray: binary indicator matrix
            float: mid point of the sigmoid function
    """
    dataset = np.array(dataset)
    scData_log = np.log(np.add(dataset, 1))
    log_mid_point = np.percentile(scData_log, percentile)
    if rate_per_gene:
        gene_mean_expressions = np.mean(scData_log, axis=0)
        print('gene_mean_expressions: ', gene_mean_expressions)
        prob_ber = np.true_divide(1, 1 + np.exp(-1*dropout_k* (gene_mean_expressions - log_mid_point)))
        binary_ind = np.array([np.random.binomial(n=1, p=prob_ber[i], size=len(scData_log)) for i in range(len(scData_log[0]))]).T
    else:
        prob_ber = np.true_divide(1, 1 + np.exp(-1*dropout_k* (scData_log - log_mid_point)))
        binary_ind = np.random.binomial(n=1, p=prob_ber)
    return binary_ind, log_mid_point


def discretize_data(data):
    """
    Discretize the data into counts.

    Args:
        data: input data

    Returns:
        np.ndarray: count matrix
    """
    return np.random.poisson(data)


def apply_noise_to_counts(
    sim: object,
    expr: list,
    outlier_prob: float = 0.01,
    outlier_mean: float = 1.0,
    outlier_scale: float = 1.0,
    lib_mean: float = 8.0,
    lib_scale: float = 0.3,
    dropout_k: float = 10.0,
    dropout_percentile: float = 80.0
):
    """
    Apply noise to the expression data.

        - Outlier genes: genes with expression values that are outliers
        - Library size: scaling factor to match random library size
        - Dropout: dropout genes by low expression
        - Conversion to UMI counts

    Args:
        sim: sergio object
        expr: expression data
        outlier_prob: probability of an outlier
        outlier_mean: mean of the outlier
        outlier_scale: scale of the outlier
        lib_mean: mean of the library size
        lib_scale: scale of the library size
        dropout_k: parameter of the dropout
        dropout_percentile: percentile of the dropout

    Returns:
        np.ndarray: count matrix with noise applied
    """
    expr_O = sim.outlier_effect(
        expr, outlier_prob, outlier_mean, outlier_scale)
    # print("expr_O shape", np.shape(expr_O))
    # print(len(expr_O), len(expr_O[0]), len(expr_O[0][0]))
    libfactors, expr_O_L = sim.lib_size_effect(expr_O, lib_mean, lib_scale)
    # print("expr_O_L shape", np.shape(expr_O_L))
    # print(len(expr_O_L), len(expr_O_L[0]), len(expr_O_L[0][0]))
    # print([len(expr_O_L[0][i]) for i in range(len(expr_O_L[0]))])
    binary_ind = sim.dropout_indicator(expr_O_L, dropout_k, dropout_percentile)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
    return count_matrix


def create_anndata(count_matrix: list, perturbation_list: list):
    """
    Turn the synthetically created dataset into the scanpy/anndata file format.
    """
    count_matrix = np.array(count_matrix).T
    obs_df = pd.DataFrame(
        [[pert] for pert in perturbation_list], columns=['perturbation'])
    var_df = pd.DataFrame(
        index=[f'gene-{i}' for i in range(len(count_matrix[0]))])

    adata = AnnData(
        X=count_matrix,
        var=var_df,
        obs=obs_df
    )

    scpi.adata.get_perturb_labels(adata)
    return adata


class Sergio_KD():
    """
    Class to run the SERGIO-KD pipeline.

    1. Generate input files for SERGIO
    2. Run the simulation
    3. Apply noise to the counts
    4. Combine to anndata object

    Data is saved in the data directory:
    data/
    ├─ raw_GRN_files/
    │   ├─ base_GRN.csv
    │   └─ base_MR.csv
    ├─ perturb_GRN_files/
    │   ├─ gene_0_perturbed_GRN.csv
    │   ├─ gene_0_perturbed_MR.csv
    |   └─ ...
    ├─ expression_files/
    │   ├─ gene_0_perturbed_replicate0_expr.npy
    │   ├─ gene_0_perturbed_replicate1_expr.npy
    |   └─ ...
    └─ output_files/
        ├─ sim_output_out_filename.h5ad
        └─ graph_out_filename.csv


    - /raw_GRN_dir: base GRN and MR files
    - /perturb_GRN_dir: perturbed GRN and MR files
    - /expression_dir: expression files (direct output of SERGIO)
    - /output_dir: output files

    Args:
        data_dir: directory to save the data
        n_genes: number of genes
        n_samples: number of samples
        n_reps: number of replicates (to be implemented)
        noise_amp: noise amplitude
        output_dir: output directory
    """

    def __init__(
            self,
            data_dir: str = './data/',
            num_genes: float = 10,
            number_sc: int = 200,
            noise_amp: float = 0.8,
            decays: float = 0.8,
            hill_coeff: float = 2,
            MR_low: float = 3,
            MR_high: float = 4,
            Hill_K: float = 4,
            KD_percent: float = 0.1,
            apply_noise: bool = False,
            apply_outliers: bool = False,
            apply_lib_size: bool = False,
            apply_dropout: bool = False,
            apply_umi_conversion: bool = False,
            dropout_k: float = 10,
            dropout_percentile: float = 80,
            edge_den: float | None = None,
            out_filename: str | None = None,
            rate_per_gene: bool = False
    ):

        # GRN information files
        self.data_dir = data_dir
        self.raw_GRN_dir = 'raw_GRN_files'
        self.perturbed_GRN_dir = 'perturb_GRN_files'
        self.expression_dir = 'expression_files'
        self.output_dir = 'output_files'

        # Simulation Parameters
        self.num_genes = num_genes   # number of genes
        self.number_bins = 1     # total number of distinct cell types
        self.number_sc = number_sc    # total number of cells per cell type, default 300
        self.noise_params = noise_amp   # noise parameters, default 1
        self.noise_type = 'dpd'  # noise type, default dpd
        self.decays = decays       # decay rate, default 0.8
        self.sampling_state = 15  # sampling state - timesteps btw sampling, default 15
        self.dt = 0.1            # time step, default 0.01

        # GRN parameters
        self.hill_coeff = hill_coeff
        self.hill_K_low = Hill_K
        self.hill_K_high = Hill_K

        # Expression / Perturbation parameters
        self.MR_low = MR_low
        self.MR_high = MR_high
        self.KD_percent = KD_percent

        # Noise parameters
        # ideally sample agains real datset for a baseline
        self.apply_noise = apply_noise
        self.apply_outliers = apply_outliers
        self.outlier_prob = 0.01     # probability of outlier genes
        # mean of the lognormal distribution from which the outlier scaling factor is sampled
        self.outlier_mean = 1
        self.outlier_scale = 1       # std
        # mean of the lognormal distribution from which the library scaling factor is sampled
        self.apply_lib_size = apply_lib_size
        self.lib_mean = 8
        self.lib_scale = 0.3         # std
        self.apply_dropout = apply_dropout
        self.dropout_k = dropout_k  # parameter of the dropout sigmoid function
        self.dropout_percentile = dropout_percentile  # percentile of the dropout distribution
        self.apply_umi_conversion = apply_umi_conversion
        self.rate_per_gene = rate_per_gene # apply dropout rate per gene

        self.adj_mat = None

        # create directories
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        for dir in [self.raw_GRN_dir, self.perturbed_GRN_dir, self.expression_dir, self.output_dir]:
            if not os.path.exists(os.path.join(self.data_dir, dir)):
                os.mkdir(os.path.join(self.data_dir, dir))

        if out_filename is None:
            if edge_den is None:
                raise ValueError(
                    'Please provide edge density or output filename')
            self.out_filename = 'q_' + \
                str(self.noise_params)+'_edgeden_' + \
                str(edge_den)+'_noise_'+str(self.apply_noise)
            print(self.out_filename)
        else:
            self.out_filename = out_filename

    def load_graph(self, file):
        """
        Load a graph from a file.
        """
        file = os.path.join(self.data_dir, self.raw_GRN_dir,
                            'graph_'+self.out_filename+'.csv')
        adj_mat = np.loadtxt(file, delimiter=',')
        self.adj_mat = adj_mat

    def gen_inputfiles(self):
        """
        Generate input files for SERGIO:

        1. Base GRN and MR files using graph and expression parameters
        2. Perturbed GRN and MR files by knocking out each gene once

        Returns:
            None
        """
        base_grn_path = os.path.join(
            self.data_dir, self.raw_GRN_dir, 'base_GRN.csv')
        base_mr_path = os.path.join(
            self.data_dir, self.raw_GRN_dir, 'base_MR.csv')

        generate_GRN_file(
            self.adj_mat,
            base_grn_path,
            hill_coefficient=self.hill_coeff,
            hill_K_low=self.hill_K_low,
            hill_K_high=self.hill_K_high
        )
        generate_MR_expression_file(
            self.num_genes,
            base_grn_path,
            base_mr_path,
            MR_low=self.MR_low,
            MR_high=self.MR_high,
        )

        np.savetxt(os.path.join(self.data_dir, self.raw_GRN_dir, 'graph.csv'),
                   self.adj_mat.astype(int), delimiter=',', fmt='%d')

    def run_simulation(self):
        """
        Run the simulation for each perturbation.
        First runs simulation on unperturbed set to compute mean expression, then again for each perturbation.
        Saves the expression data to files.

        Returns:
            None
        """

        out_dir = os.path.join(self.data_dir, self.expression_dir)
        empty_dir(out_dir)  # remove previous output files

        print('Running initial simulation with base GRN to compute mean expression values.')

        run_simulation(
            [os.path.join(self.data_dir, self.raw_GRN_dir)],
            out_dir,
            number_genes=self.num_genes,
            number_bins=1,
            number_sc=self.number_sc,
            noise_params=self.noise_params,
            noise_type=self.noise_type,
            decays=self.decays,
            sampling_state=self.sampling_state,
            dt=self.dt,
            number_replicates=5,
            hill_coefficient=self.hill_coeff,
        )

        out_file = os.listdir(out_dir)[0]

        half_responses = get_mean_expressions(os.path.join(out_dir, out_file))
        print('Computed Half-Response Values: ', half_responses)

        empty_dir(out_dir)  # remove previous output files

        print('Re-running unperturbed simulation with updated half-response-values to compute mean expression values.')

        run_simulation(
            [os.path.join(self.data_dir, self.raw_GRN_dir)],
            out_dir,
            number_genes=self.num_genes,
            number_bins=1,
            number_sc=self.number_sc,
            noise_params=self.noise_params,
            noise_type=self.noise_type,
            decays=self.decays,
            sampling_state=self.sampling_state,
            dt=self.dt,
            number_replicates=5,
            hill_coefficient=self.hill_coeff,
            half_responses=half_responses
        )

        mean_exprs = get_mean_expressions(os.path.join(out_dir, out_file))
        print('mean epression values: ', mean_exprs)

        print('Creating perturbed GRN and MR files.')
        base_grn_path = os.path.join(
            self.data_dir, self.raw_GRN_dir, 'base_GRN.csv')
        base_mr_path = os.path.join(
            self.data_dir, self.raw_GRN_dir, 'base_MR.csv')
        generate_perturbation_files(self.num_genes, base_grn_path, base_mr_path, output_dir=os.path.join(
            self.data_dir, self.perturbed_GRN_dir), mean_exprs=mean_exprs, KD_percent=self.KD_percent)

        empty_dir(out_dir)  # remove previous output files

        print('Running full simulation with unperturbed and perturbed GRN.')

        # run simulation with perturbed GRN
        run_simulation(
            [os.path.join(self.data_dir, self.raw_GRN_dir)],
            out_dir,
            number_genes=self.num_genes,
            number_bins=1,
            number_sc=self.number_sc,
            noise_params=self.noise_params,
            noise_type=self.noise_type,
            decays=self.decays,
            sampling_state=self.sampling_state,
            dt=self.dt,
            number_replicates=20,
            hill_coefficient=self.hill_coeff,
            half_responses=mean_exprs,
        )

        run_simulation(
            [os.path.join(self.data_dir, self.perturbed_GRN_dir)],
            out_dir,
            number_genes=self.num_genes,
            number_bins=1,
            number_sc=self.number_sc,
            noise_params=self.noise_params,
            noise_type=self.noise_type,
            decays=self.decays,
            sampling_state=self.sampling_state,
            dt=self.dt,
            number_replicates=5,
            hill_coefficient=self.hill_coeff,
            half_responses=mean_exprs,
        )

    def apply_noise_to_counts_mod(
        self,
        sim: object,
        expr: list,
    ):
        """
        Apply noise to the expression data.

            - Outlier genes: genes with expression values that are outliers
            - Library size: scaling factor to match random library size
            - Dropout: dropout genes by low expression
            - Conversion to UMI counts

        Args:
            sim: sergio object
            expr: expression data

        Returns:
            np.ndarray: count matrix with noise applied
        """
        # 1. Outlier genes
        if self.apply_outliers:
            expr_O = sim.outlier_effect(
                expr, self.outlier_prob, self.outlier_mean, self.outlier_scale)
        else:
            expr_O = expr
        # 2. Library size
        if self.apply_lib_size:
            libfactors, expr_O_L = sim.lib_size_effect(
                expr_O, self.lib_mean, self.lib_scale)
        else:
            expr_O_L = expr_O
        # 3. Dropout
        if self.apply_dropout:
            binary_ind, lim = sample_dropout(
                expr_O_L, self.dropout_k, self.dropout_percentile, rate_per_gene=self.rate_per_gene)
            expr_O_L_D = np.multiply(binary_ind, expr_O_L)
        else:
            expr_O_L_D = expr_O_L
            lim = None
        # 4. Conversion to UMI counts
        if self.apply_umi_conversion:
            count_matrix = discretize_data(expr_O_L_D)
        else:
            count_matrix = expr_O_L_D
        
        return count_matrix, lim

    def combine_expr(self, normalize=False, shifted_log=False):
        """
        Combine the expression files into a single AnnData object.
        Apply noise to the counts if enabled.
        """
        sim = sergio(self.num_genes, self.number_bins, self.number_sc, self.noise_params,
                     self.noise_type, self.decays, sampling_state=self.sampling_state, dt=self.dt)

        perturbations = []
        expr_files = os.listdir(os.path.join(
            self.data_dir, self.expression_dir))
        expr_files.sort()
        perturbations = [
            'gene-'+file.split('_')[-4] if 'perturbed' in file else 'non-targeting' for file in expr_files]
        print('expression_files: ', expr_files[:3])
        print('perturbations: ', perturbations[:3])

        # concatenate the individual arrays per perturbation (and cell type) to one array
        # and apply noise if enabled

        complete_counts = []
        complete_perturbations = []
        for expr_file, perturbation in zip(expr_files, perturbations):    
            counts = np.load(os.path.join(self.data_dir, self.expression_dir, expr_file))  # shape: 1, 10, 200  ---- 1, n_genes, n_samp
            complete_counts.append(np.concatenate(counts, axis = 1))
            complete_perturbations.extend([perturbation]*np.shape(counts)[2])
        
        print('complete_counts: ',np.shape(complete_counts))
        print('complete_perturbations: ',np.shape(complete_perturbations))

        
        counts_cat = np.concatenate(complete_counts, axis=1)
        if self.apply_noise:
            # apply noise according to the parameters
            counts_noise, lim = self.apply_noise_to_counts_mod(sim, counts_cat)
        else:
            counts_noise = counts_cat
        adata = create_anndata(counts_noise, complete_perturbations)

        if normalize:
            mean_genes_per_cell = np.mean(
                np.sum(adata[adata.obs['perturbation'] == 'non-targeting'].X, axis=1))
            sc.pp.normalize_total(adata, target_sum=mean_genes_per_cell)
        if shifted_log:
            sc.pp.log1p(adata)

        print("")
        print(' Created AnnData object: ')
        print(adata)
        print('gene names: ', [n for n in adata.var_names])
        print(adata.obs['perturbation'].value_counts())

        out_file = os.path.join(
            self.data_dir, self.output_dir, 'sim_output_'+self.out_filename+'.h5ad')
        print('Writing AnnData object to: ', out_file)
        sc.write(out_file, adata)

        # save graph adjacency matrix to file
        file = os.path.join(self.data_dir, self.output_dir,
                            'graph_'+self.out_filename+'.csv')
        print('Saving graph to: ', file)
        np.savetxt(file, self.adj_mat.astype(int), delimiter=',', fmt='%d')
        # save the lim value for dropout
        if self.apply_dropout and self.apply_noise:
            file = os.path.join(self.data_dir, self.output_dir,
                                'lim_'+self.out_filename+'.txt')
            print('Saving lim value to: ', file)
            np.savetxt(file, [lim], fmt='%f')
