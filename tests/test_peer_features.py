import pandas as pd
import numpy as np
import pytest
from ml_backend.data_loader import add_peer_based_features

def test_peer_features_permno_index():
    # Create dummy data
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"])
    permnos = [1, 2, 1, 2]
    data = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0, 4.0]
    }, index=pd.MultiIndex.from_arrays([dates, permnos], names=["date", "permno"]))

    # Create embeddings with only permno index
    # permno 1: [0, 0], permno 2: [1, 0] -> dist = 1
    embeddings = pd.DataFrame({
        "emb1": [0.0, 1.0],
        "emb2": [0.0, 0.0]
    }, index=pd.Index([1, 2], name="permno"))

    # Expected behavior:
    # Date 2020-01-01:
    # p1(0,0), p2(1,0). dist=1. bw=1. 
    # w_12 = exp(-1/2) = 0.606
    # w_21 = exp(-1/2) = 0.606
    # norm: w_12 = 1, w_21 = 1 (since only 1 peer)
    # peer_feat_1 = val_2 = 2.0
    # peer_feat_2 = val_1 = 1.0
    
    # Date 2020-02-01:
    # same embeddings.
    # peer_feat_1 = val_2 = 4.0
    # peer_feat_2 = val_1 = 3.0
    
    res = add_peer_based_features(data, embeddings)
    
    assert "feature_peer" in res.columns
    
    # Check 2020-01-01
    # p1 should have p2's value
    assert res.loc[("2020-01-01", 1), "feature_peer"] == pytest.approx(2.0)
    assert res.loc[("2020-01-01", 2), "feature_peer"] == pytest.approx(1.0)
    
    # Check 2020-02-01
    assert res.loc[("2020-02-01", 1), "feature_peer"] == pytest.approx(4.0)
    assert res.loc[("2020-02-01", 2), "feature_peer"] == pytest.approx(3.0)


def test_peer_features_time_varying_index():
    # Create dummy data
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"])
    permnos = [1, 2, 1, 2]
    data = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0, 4.0]
    }, index=pd.MultiIndex.from_arrays([dates, permnos], names=["date", "permno"]))

    # Create embeddings with date and permno
    # 2020-01-01: p1=[0,0], p2=[1,0] (dist=1, bw=1, w=exp(-0.5))
    # 2020-02-01: p1=[0,0], p2=[2,0] (dist=2, bw=2)
    
    embeddings = pd.DataFrame({
        "emb1": [0.0, 1.0, 0.0, 2.0],
        "emb2": [0.0, 0.0, 0.0, 0.0]
    }, index=data.index) # same index

    res = add_peer_based_features(data, embeddings)
    
    # 2020-01-01 should be same as before
    assert res.loc[("2020-01-01", 1), "feature_peer"] == pytest.approx(2.0)
    assert res.loc[("2020-01-01", 2), "feature_peer"] == pytest.approx(1.0)
    
    # 2020-02-01
    # p1(0), p2(2). dist=2. bw=2. 
    # w = exp(-4 / (2*4)) = exp(-0.5)
    # normalized: 1.0 each.
    # peer_feat_1 = val_2 = 4.0
    # peer_feat_2 = val_1 = 3.0
    assert res.loc[("2020-02-01", 1), "feature_peer"] == pytest.approx(4.0)
    assert res.loc[("2020-02-01", 2), "feature_peer"] == pytest.approx(3.0)
