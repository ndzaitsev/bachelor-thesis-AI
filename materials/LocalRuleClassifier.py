import jpype
import jpype.imports
import numpy as np
import tempfile
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter

# internal config
jar_path = "/LORD-master/lord.jar"
concrete_class_name = "rl.eg.Lord"

class InfoBase:
    def __init__(self, learner):
        self.learner = learner
        self.selector_nlists = learner.getSelectorNlists()
        self.constructing_selectors = list(learner.getConstructingSelectors())
        self.selector_id_records = learner.getSelectorIDRecords()
        self.class_ids = list(learner.getClassIDs())
        self.RuleSearcher = jpype.JClass("rl.RuleSearcher")
        self.INlist = jpype.JClass("rl.INlist")

    def support_count(self, selector_ids):
        nlist_array = jpype.JArray(self.INlist)([self.selector_nlists[i] for i in selector_ids])
        return self.RuleSearcher.calculate_nlist_direct(nlist_array).supportCount()

class LocalRuleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, metric="MESTIMATE", metric_arg=0.1):
        self.metric = metric
        self.metric_arg = metric_arg

        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[jar_path])

        self.RuleLearnerClass = jpype.JClass(concrete_class_name)
        self.METRIC_TYPES = jpype.JClass("evaluations.HeuristicMetricFactory$METRIC_TYPES")
        self.IntHolder = jpype.JClass("rl.IntHolder")
        self.learner = self.RuleLearnerClass()

    def _write_temp_csv(self, X, y):
        fd, path = tempfile.mkstemp(suffix=".csv", text=True)
        with os.fdopen(fd, 'w') as tmp:
            for xi, yi in zip(X, y):
                tmp.write(",".join(map(str, list(xi) + [str(yi)])) + "\n")
        return path

    def fit(self, X, y):
        file_path = self._write_temp_csv(X, y)
        return self.fit_csv(file_path, y)

    def fit_csv(self, file_path, y=None):
        metric_enum = getattr(self.METRIC_TYPES, self.metric)
        self.learner.fetch_information(file_path)
        self.learner.learning(metric_enum, float(self.metric_arg))

        class_ids = list(self.learner.getClassIDs())
        self.train_file_ = file_path

        if y is not None:
            unique_labels = sorted(np.unique(y), reverse=True)
            self.class_id_to_label_ = {cid: unique_labels[i] for i, cid in enumerate(class_ids)}
            self.classes_ = np.array(unique_labels)
        else:
            self.class_id_to_label_ = {cid: str(cid) for cid in class_ids}
            self.classes_ = np.array([str(cid) for cid in class_ids])

        return self

    def predict(self, X):
        results = []
        for row in X:
            row_str = list(map(str, row)) + ["?"]
            holder = self.IntHolder(0)
            self.learner.predict(row_str, holder)

            class_id = holder.value
            label = self.class_id_to_label_.get(class_id, self.classes_[0])
            results.append(label)

        return np.array(results)

    def get_info_base(self):
        return InfoBase(self.learner)

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {
            "metric": self.metric,
            "metric_arg": self.metric_arg
        }


