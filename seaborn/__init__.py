import seaborn.apionly as sns
import numpy as np
import matplotlib.pyplot as plt


class FacetGrid(sns.FacetGrid):
    """
    interface to seaborn FacetGrid with
    user defined axes
    """
    def __init__(self, *args, **kwargs):
        """
        create seaborn FacetGrid

        see FacetGrid documentation for further details

        Extra Parameters
        ----------------
        axes : array of plot axes, optional
            axes to plot data to.  If this is supplied, then these override
            FacetGrid.axes.
        tight_layout : bool, default=True
            if True, perform fig.tight_layout()

        """

        my_axes = kwargs.pop('axes', None)
        self._tight_layout = kwargs.pop('tight_layout', True)

        # initialize
        super(self.__class__, self).__init__(*args, **kwargs)

        if my_axes is not None:
            # close old figure
            plt.close(self.fig)

            # check passed axes
            my_axes = np.atleast_2d(my_axes)
            assert my_axes.shape[0] >= self._nrow, 'not enough rows'
            assert my_axes.shape[1] >= self._ncol, 'not enough cols'

            self.fig = my_axes[0, 0].get_figure()
            self.axes = my_axes


            # redo despine?
            if kwargs.get('despine', True):
                self.despine()


    # override FacetGrid method
    def _finalize_grid(self, axlabels):
        """Finalize the annotations and layout."""
        self.set_axis_labels(*axlabels)
        self.set_titles()
        if self._tight_layout:
            self.fig.tight_layout()





def meta_plot(x, y, data, keys=None, kwargs_func=None, **kwargs):
    """
    function to pass to sns.FacetGrid.map_dataframe

    Parameters
    ----------
    x, y : string
        keys for x and y data
    data : dataframe
    keys : list of strings, optional
        if supplied, these columns of the dataframe override **kwargs.
        Note that if values of `None` in data cause some issues.  Therefore,
        the function looks for values of `'str_None'` and replaces with `None`
    kwargs_func : function, optional
        function that takes as arguments data.iloc[0].to_dict() or
        each group of data.groupby(keys)
        and returns
        f_kwargs dictionary.  f_kwargs overrides any arguments
        in kwargs or data[keys].

    **kwargs : extra arguments to plt.plot

    Examples
    --------

    (sns.FacetGrid(df.assign(mfc='blue'), hue='z')
    .map_dataframe(meta_plot, 'x','y',keys=['mfc'])
    )

    """
    ax = plt.gca()
    if len(data) == 0 or keys is None:

        if kwargs_func is not None:
            k = kwargs_func(data.iloc[0].to_dict())
            k = dict(kwargs, **k)
        else:
            k = kwargs

        ax.plot(data[x],data[y], **k)
    else:
        #label = kwargs.pop('label',None)
        label = kwargs.get('label',None)

        for i,(v, g) in enumerate(data.groupby(keys)):
            if i==0:
                l = label
            else:
                l = None

            # replace str_None with None
            vv = [None if _ == 'str_None' else _ for _ in np.atleast_1d(v)]
            d = dict(zip(keys, vv), label=l)

            if kwargs_func is not None:
                k = kwargs_func(g.iloc[0].to_dict())
                d = dict(d, **k)

            k = dict(kwargs, **d)
            ax.plot(g[x].values, g[y].values, **k)
