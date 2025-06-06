
from typing import List
import matplotlib.pyplot as plt

from sdnist.utils import *

plt.style.use('seaborn-v0_8-deep')


class CorrelationDifferencePlot:
    def __init__(self,
                 synthetic: pd.DataFrame,
                 target: pd.DataFrame,
                 output_directory: Path,
                 features: List[str]):
        """
        Computes and plots features correlation difference between
        synthetic and target data

        Parameters
        ----------
            synthetic : pd.Dataframe
                synthetic dataset
            target : pd.Dataframe
                target dataset
            output_directory: pd.Dataframe
                path of the directory to which plots will will be saved
            features: List[str]
                List of names of features for which to compute correlation
        """
        self.syn = synthetic
        self.tar = target
        self.o_dir = output_directory
        self.o_path = Path(self.o_dir, 'kendall')
        self.features = features
        self.report_data = dict()

        self._setup()

    def _setup(self):
        if not self.o_dir.exists():
            raise Exception(f'Path {self.o_dir} does not exist. Cannot save plots')

        os.mkdir(self.o_path)

    def save(self) -> List[Path]:
        corr_df = correlation_difference(self.syn, self.tar, self.features)
        plot_paths = save_correlation_difference_plot(corr_df, self.o_path)
        self.report_data = {"correlation_difference": relative_path(save_data_frame(corr_df,
                                                                      self.o_path,
                                                                      'diff')),
                            "plot": relative_path(plot_paths[0])}
        return plot_paths


def correlations(data: pd.DataFrame, features: List[str]):
    corr_list = []

    for f_a in features:
        f_a_corr = []
        for f_b in features:
            x, y = data[f_a].values, data[f_b].values
            if f_a == f_b:
                f_a_corr.append(0)
            elif np.all(x == x[0]) or np.all(y == y[0]):
                f_a_corr.append(0)
            else:
                c_val = data[f_a].corr(data[f_b], method='kendall')
                f_a_corr.append(c_val)
        corr_list.append(f_a_corr)

    return pd.DataFrame(corr_list, columns=features, index=features)


def correlation_difference(synthetic: pd.DataFrame,
                           target: pd.DataFrame,
                           features: List[str]) -> pd.DataFrame:

    syn_corr = correlations(synthetic, features)
    tar_corr = correlations(target, features)

    diff = syn_corr - tar_corr

    return diff


def save_correlation_difference_plot(correlation_data: pd.DataFrame,
                                     output_directory: Path) -> List[Path]:
    cd = correlation_data
    cd = cd.abs()
    fig = plt.figure(figsize=(6, 6), dpi=100)
    v_max = 0.15
    plt.imshow(cd, cmap='Blues', interpolation='none', vmin=0, vmax=v_max)
    im_ratio = cd.shape[0] / cd.shape[1]
    cbar = plt.colorbar(fraction=0.04 * im_ratio)
    cbar.ax.tick_params(labelsize=10)
    plt.xticks(range(cd.shape[1]), cd.columns)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(range(cd.shape[0]), cd.index, fontsize=10)
    file_path = Path(output_directory, 'diff.jpg')
    plt.title('Correlation Diff. Between Target and Deid. Data', fontsize=12)
    fig.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    return [file_path]


