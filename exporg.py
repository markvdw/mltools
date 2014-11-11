# I want this module to help sort out experiments and experimental results. I want:
#  - To be able to run experiments with different parameters easily
#  - Automatically store the results (plots, processed files etc) of the experiment
#  - Retrieve and view experimental results and compare between different parameter settings

# To do so, I need to store the experimental parameters together with the results.
# I need to be able to initialise the code back to the final state so I can keep processing with the results later on.
# I need a viewer which allows me to sort and group by parameter settings.

import datetime
import shutil
import os

import drawtools as mldraw


class ExperimentBase(object):
    def __init__(self):
        self._time_str = str(datetime.datetime.now()).replace(' ', '_')
        self._savedir = None
        self.figlist = []

    def savefigs(self, filename="figs.pdf"):
        path = self.savedir + filename
        print path
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        mldraw.figs_to_pdf(path, self.figlist)

    def save_terminal_output(self, logger):
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        logger.logflush()
        shutil.copy(logger.filepath, self.savedir + 'log.txt')

    def save(self):
        # Just save the parameters
        f = open(self.savedir + "params.txt", 'w')
        for k, v in self.params.iteritems():
            f.write(str(k) + '\t' + str(v))
            f.write('\n')

        f.close()

    @property
    def savedir(self):
        if self._savedir is None:
            return "./results/" + self.params['name'] + self._time_str + '/'
        else:
            return self._savedir

    @savedir.setter
    def savedir(self, val):
        if val[-1] != '/':
            val += '/'
        self._savedir = val
