"""Check decision boundaries that can silently invalidate business results."""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from decision_analytics.common import model_pipeline
from decision_analytics.policies import bank_policies, evaluate_bank, select_prefix
from decision_analytics.routing import build_route, distance_matrix, route_distance, select_threshold


class BankPolicyTests(unittest.TestCase):
    def test_budget_boundary_and_stable_ties(self):
        np.testing.assert_array_equal(select_prefix([.9,.9,.8],[2.,3.,1.],5.),[0,1])
        self.assertEqual(len(select_prefix([.9,.8],[2.,1.],0.)),0)

    def test_infeasible_leading_item_does_not_skip_in_prefix_heuristic(self):
        # The documented rule is a sorted prefix, not a greedy knapsack fill.
        self.assertEqual(len(select_prefix([.9,.8],[10.,1.],5.)),0)

    def test_negative_expected_profit_is_excluded(self):
        rows=bank_policies([.01,.8],[1.,1.],100.,5.,1.,5.)
        np.testing.assert_array_equal(rows['Duration-aware ranking'],[1])

    def test_predicted_budget_can_pass_while_actual_fails(self):
        result=evaluate_bank([0,1],[.7,.8],[2.,2.],[1,0],[3.,4.],100.,1.,2.,5.)
        self.assertTrue(result['predicted_within_budget'])
        self.assertFalse(result['actual_within_budget'])
        self.assertAlmostEqual(result['actual_overrun_pct'],40.)
        self.assertAlmostEqual(result['historical_simulated_profit'],84.)

    def test_actual_outcomes_cannot_enter_selector(self):
        p,t=np.array([.9,.4,.7]),np.array([1.,2.,1.])
        first=bank_policies(p,t,100.,1.,1.,2.)
        for outcomes in [[1,0,1],[0,1,0]]:
            _=evaluate_bank(first['Probability ranking'],p,t,outcomes,[100.,1.,100.],100.,1.,1.,2.)
            np.testing.assert_array_equal(bank_policies(p,t,100.,1.,1.,2.)['Probability ranking'],first['Probability ranking'])

    def test_invalid_durations_rejected(self):
        for minutes in [[0.],[-1.],[np.nan]]:
            with self.assertRaises(ValueError): select_prefix([.5],minutes,1.)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.d=distance_matrix(np.array([[0.,.01],[0.,.02],[0.,.03]]),depot=(0.,0.))

    def test_symmetric_distance_and_round_trip(self):
        np.testing.assert_allclose(self.d,self.d.T)
        np.testing.assert_allclose(np.diag(self.d),0.)
        route=build_route(self.d,5.)
        self.assertEqual(route[0],0);self.assertEqual(route[-1],0)
        self.assertEqual(len(set(route[1:-1])),len(route[1:-1]))
        self.assertLessEqual(route_distance(route,self.d),5.+1e-8)

    def test_return_to_depot_is_reserved(self):
        # The first candidate is 1.11km away: a 1.5km one-way-looking budget is insufficient.
        self.assertEqual(build_route(self.d,1.5),[0,0])

    def test_empty_candidate_set_and_zero_budget(self):
        self.assertEqual(build_route(self.d,0.),[0,0])
        self.assertEqual(build_route(self.d,10.,[]),[0,0])

    def test_threshold_search_requires_no_outcomes(self):
        threshold,route,rows=select_threshold([.8,.2,.1],self.d,5.,[.1,.5,.9])
        self.assertIn(threshold,[.1,.5,.9])
        self.assertLessEqual(route_distance(route,self.d),5.+1e-8)
        self.assertTrue(all('historical_simulated_net' not in row for row in rows))


class PreprocessingTests(unittest.TestCase):
    def test_unseen_category_and_training_only_imputation(self):
        tr=pd.DataFrame({'number':[1.,2.,3.,np.nan], 'kind':['a','b','a','b']})
        model=model_pipeline(tr,LogisticRegression())
        model.fit(tr,[0,1,0,1])
        te=pd.DataFrame({'number':[100000.,np.nan],'kind':['unseen','a']})
        pred=model.predict_proba(te)
        self.assertTrue(np.isfinite(pred).all())
        imp=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
        self.assertEqual(float(imp.statistics_[0]),2.)


if __name__=='__main__':
    unittest.main()
