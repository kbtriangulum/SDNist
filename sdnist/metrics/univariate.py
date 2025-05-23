from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from sdnist.report import Dataset
from sdnist.strs import *
from sdnist.utils import *

plt.style.use('seaborn-v0_8-deep')


def l1(pk: List[int], qk: List[int]):
    pk_max = max(pk)
    qk_max = max(qk)

    if pk_max == 0:
        pk_n = [0 for p in pk]
    else:
        pk_n = [p/pk_max for p in pk]

    if qk_max == 0:
        qk_n = [0 for q in qk]
    else:
        qk_n = [q/qk_max for q in qk]

    div = sum([abs(p-q) for p, q in zip(pk_n, qk_n)])

    return div



class UnivariatePlots:
    def __init__(self,
                 synthetic: pd.DataFrame,
                 target: pd.DataFrame,
                 dataset: Dataset,
                 output_directory: Path,
                 challenge: str = CENSUS,
                 worst_univariates_to_display: Optional[int] = None):
        """
        Computes and creates univariate distribution plots of the worst
        performing variables in synthetic data

        Parameters
        ----------
            synthetic : pd.Dataframe
                synthetic dataset
            target : pd.Dataframe
                target dataset
            dataset : Dataset
                dataset object
            output_directory: pd.Dataframe
                path of the directory to which plots will be saved
            challenge: str
                For which challenge type to compute univariates for, CENSUS or TAXI
            n : pd.Dataframe
                n worst performing univariates to save plots for
        """
        self.syn = synthetic
        self.tar = target

        self.schema = dataset.schema
        self.dataset = dataset
        self.bin_mappings = dataset.bin_mappings
        self.o_dir = output_directory
        self.out_path = Path(self.o_dir, 'univars')
        self.worst_univariates_to_display = worst_univariates_to_display
        self.challenge = challenge
        self.feat_data = dict()
        self._setup()

        self.div_data = None  # feature divergence data
        self.uni_counts = dict()  # univariate counts of target and synthetic data

    def _setup(self):
        if not self.o_dir.exists():
            raise Exception(f'Path {self.o_dir} does not exist. Cannot save plots')
        os.mkdir(self.out_path)

    def report_data(self, level=2):
        return {"divergence": relative_path(save_data_frame(self.div_data,
                                                            self.out_path,
                                                            'div'),
                                            level=level),
                "counts": self.uni_counts}

    def save(self, level=2) -> Dict:
        if self.challenge == CENSUS:
            ignore_features = ['YEAR']
        elif self.challenge == TAXI:
            ignore_features = ['pickup_community_area', 'shift', 'company_id']
        else:
            raise Exception(f'Invalid Challenge Name: {self.challenge}. '
                            f'Unable to save univariate plots')
        # divergence dataframe
        div_df = divergence(self.syn,
                            self.tar,
                            self.schema, ignore_features)
        self.div_data = div_df
        # select 3 features with worst divergence
        # div_df = div_df.head(3)

        self.save_distribution_plot(self.dataset,
                                    self.syn,
                                    self.tar,
                                    div_df[FEATURE].tolist(),
                                    self.out_path,
                                    level=level)
        return self.feat_data

    def save_distribution_plot(self,
                               dataset: Dataset,
                               synthetic: pd.DataFrame,
                               target: pd.DataFrame,
                               features: List,
                               output_directory: Path,
                               level=2):
        ds = dataset
        o_path = output_directory
        bar_width = 0.4
        saved_file_paths = []
        INDP = 'INDP'
        INDP_CAT = "INDP_CAT"
        o_tar = ds.target_data.loc[target.index]
        o_syn = ds.c_synthetic_data.loc[synthetic.index]
        schema = ds.schema

        for i, f in enumerate(features):
            self.uni_counts[f] = dict()
            if f == INDP and INDP_CAT in target.columns.tolist():
                all_sectors = o_tar[INDP_CAT].unique().tolist()
                set(all_sectors).update(set(o_syn[INDP_CAT].unique().tolist()))
                selected = []
                for s in all_sectors:
                    if s == 'N':
                        continue
                    st_df = o_tar[o_tar[INDP_CAT].isin([s])].copy()
                    st_df.loc[:, f] = pd.to_numeric(st_df[f]).astype(int)
                    ss_df = o_syn[o_syn[INDP_CAT].isin([int(s), s])]
                    unique_ind_codes = st_df[f].unique().tolist()
                    set(unique_ind_codes).update(set(ss_df[f].unique().tolist()))
                    unique_ind_codes = list(unique_ind_codes)
                    val_df = pd.DataFrame(unique_ind_codes, columns=[f])
                    val_df[f] = val_df.astype(str)

                    t_counts_df = st_df.groupby(by=f)[f].size().reset_index(name='count_target')
                    s_counts_df = ss_df.groupby(by=f)[f].size().reset_index(name='count_deidentified')
                    t_counts_df[f] = t_counts_df[f].astype(str)
                    s_counts_df[f] = s_counts_df[f].astype(str)

                    merged = pd.merge(left=val_df, right=t_counts_df, on=f, how='left')\
                        .fillna(0)
                    merged = pd.merge(left=merged, right=s_counts_df, on=f, how='left')\
                        .fillna(0)
                    div = l1(pk=merged['count_target'], qk=merged['count_deidentified'])

                    selected.append([merged, div, s])
                selected = sorted(selected, key=lambda l: l[1], reverse=True)

                for j, data in enumerate(selected):
                    merged = data[0]
                    div = data[1]
                    s = data[2]
                    merged = merged.sort_values(by=f)
                    x_axis = np.arange(merged.shape[0])
                    plt.figure(figsize=(8, 3), dpi=100)
                    plt.bar(x_axis - 0.2, merged['count_target'], width=bar_width, label='Target')
                    plt.bar(x_axis + 0.2, merged['count_deidentified'], width=bar_width, label='Deidentified')
                    plt.xlabel('Feature Values', fontsize=10)
                    plt.ylabel('Record Counts', fontsize=10)
                    plt.gca().set_xticks(x_axis, merged[f].values.tolist())
                    plt.legend(loc='upper right', fontsize=10)
                    if merged.shape[0] > 30:
                        plt.xticks(fontsize=6, rotation=90)
                    else:
                        plt.xticks(fontsize=8, rotation=45)
                    plt.tight_layout()
                    title = f'Industries in Industry Category ' \
                            f'{dataset.data_dict["INDP_CAT"]["values"][str(s)]}'
                    plt.title(title,
                              fontdict={'fontsize': 12})

                    file_path = Path(o_path, f'indpcat_{s}.jpg')
                    plt.savefig(file_path, bbox_inches='tight')

                    plt.close()
                    self.uni_counts[f][f"Industry Category {s}"] = {
                        "divergence": div,
                        "counts": relative_path(save_data_frame(merged,
                                                o_path,
                                                f"indp_cat_{s}"),
                                                level=level),
                        "plot": relative_path(file_path, level=level)
                    }
                    # if j < 2:
                    saved_file_paths.append(file_path)

                    self.feat_data[title] = {
                        "path": ''
                    }
            else:
                plt.figure(figsize=(8, 3), dpi=100)
                file_path = Path(o_path, f'{f}.jpg')
                merged = self.count_values(f)

                title = f"{f}: {dataset.data_dict[f]['description']}"
                self.uni_counts[f] = {
                    "divergence": self.div_data[self.div_data[FEATURE] == f][DIVERGENCE].values[0],
                    "counts": relative_path(save_data_frame(merged.copy(),
                                                            o_path,
                                                            f'{f}'),
                                            level=level),
                    "plot": relative_path(file_path, level)
                }

                if self.worst_univariates_to_display is None \
                        or i < self.worst_univariates_to_display:
                    merged = self.separate_large_count_value(f, merged, title)


                x_axis = np.arange(merged.shape[0])
                plt.bar(x_axis - 0.2, merged['count_target'], width=bar_width, label='Target')
                plt.bar(x_axis + 0.2, merged['count_deidentified'], width=bar_width, label='Deidentified')
                plt.xlabel('Feature Values', fontsize=10)
                plt.ylabel('Record Counts', fontsize=10)
                vals = merged[f].values.tolist()


                vals = [str(v) for v in vals]
                if "-1" in vals:
                    idx = vals.index("-1")
                    vals[idx] = "N"

                plt.gca().set_xticks(x_axis, vals)
                plt.legend(loc='upper right', fontsize=10)
                plt.xticks(fontsize=8, rotation=45)
                plt.yticks(fontsize=10)
                plt.tight_layout()

                plt.title(title, fontdict={'fontsize': 12})

                plt.savefig(Path(o_path, f'{f}.jpg'), bbox_inches='tight')
                plt.close()

                if i < self.worst_univariates_to_display:
                    self.feat_data[title]['path'] = file_path
                if i >= self.worst_univariates_to_display:
                    break
        return saved_file_paths

    def count_values(self, feature: str) -> pd.DataFrame:
        t, d, f = self.tar, self.syn, feature
        values = sorted(set(t[f].unique()).union(d[f].unique()))
        t_counts = t[f].value_counts().reindex(values, fill_value=0)
        d_counts = d[f].value_counts().reindex(values, fill_value=0)

        merged = pd.DataFrame({f: values,
                               'count_target': t_counts,
                               'count_deidentified': d_counts}).fillna(0)

        if f in self.bin_mappings:
            merged[f] = merged[f].map(self.bin_mappings[f]).fillna(merged[f])
        return merged

    def separate_large_count_value(self,
                                   feature: str,
                                   merged: pd.DataFrame, title: str):
        self.feat_data[title] = {}
        s_merged = merged.sort_values(by='count_target', ascending=False)
        if s_merged.shape[0] > 1 and s_merged['count_target'].iloc[0] >= \
                s_merged['count_target'].iloc[1] * 3:
            val = s_merged[feature].iloc[0]
            self.feat_data[title] = {
                "excluded": {
                    "feature_value": s_merged[feature].iloc[0],
                    "target_counts": int(
                        s_merged['count_target'].iloc[0]),
                    "deidentified_counts": int(
                        s_merged['count_deidentified'].iloc[0])
                }
            }
            # remove val from merged
            merged = merged[~merged[feature].isin([val])]
        return merged




def divergence(synthetic: pd.DataFrame,
               target: pd.DataFrame,
               schema: Dict[str, any],
               ignore_features: Optional[List[str]] = None):
    if not ignore_features:
        ignore_features = []

    div_data = []  # divergence data
    tfeats = target.columns.tolist()
    sfeats = synthetic.columns.tolist()
    for var, var_schema in schema.items():
        if var not in tfeats or var not in sfeats:
            continue
        if var in ignore_features:
            continue
        values = set(target[var].unique().tolist()).union(synthetic[var].unique().tolist())
        values = sorted(values)
        val_df = pd.DataFrame(values, columns=[var])

        s_counts_df = synthetic.groupby(by=var)[var].size().reset_index(name=COUNT)
        t_counts_df = target.groupby(by=var)[var].size().reset_index(name=COUNT)

        s_counts_df = pd.merge(left=val_df, right=s_counts_df, on=var, how='left', sort=True)\
            .fillna(0)
        t_counts_df = pd.merge(left=val_df, right=t_counts_df, on=var, how='left', sort=True)\
            .fillna(0)

        div = l1(pk=s_counts_df[COUNT], qk=t_counts_df[COUNT])
        div_data.append([var, div])

    return pd.DataFrame(div_data, columns=[FEATURE, DIVERGENCE])\
        .sort_values(by=DIVERGENCE, ascending=False)
