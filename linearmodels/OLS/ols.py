import numpy as np
import patsy as pt


class LinearModel:
    """
    A linear model.
    """
    def __init__(self):
        pass
        

    def fit(self, y, x, add_intercept=True):
        """
        Fit the linear model.
        
        Returns
        -------
        FittedLinearModel
            The fitted linear model.
        """
        if add_intercept:
            x = np.c_[np.ones(len(x)), x]
        
        q, r = np.linalg.qr(x, mode='reduced')
        return FittedLinearModel(self, np.linalg.solve(r, q.T @ y))
    
    def fit_from_formula(self, formula, data):
        """
        Fit the linear model from a formula.
        
        Parameters
        ----------
        formula : str
            The formula.
        data : DataFrame
            The data.
        
        Returns
        -------
        FittedLinearModel
            The fitted linear model.
        """
        y, x = pt.dmatrices(formula, data) # patsy adds intercept by default
        return self.fit(y, x, add_intercept=False)
    

class FittedLinearModel:
    """
    A fitted linear model.

    Parameters
    ----------
    model : LinearModel
        The linear model object.
    params : ndarray
        The estimated parameters.
    """
    def __init__(self, model, params):
        self.model = model
        self.params = params

    @property
    def fittedvalues(self):
        return self.predict(self.model.x)
    
    @property
    def resid(self):
        return self.model.y - self.fittedvalues
    
    @property
    def model(self):
        return self._model
    
    @property
    def params(self):
        return self._params

    def predict(self, x):
        return x @ self.params
