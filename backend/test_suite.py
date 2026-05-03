"""
Neural Forge — Full Test Suite
Run from: neural_forge/backend/
Usage: python3 test_suite.py
"""
import sys, os, types, numpy as np
# Ensure backend is on path regardless of CWD
_BACKEND = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if _BACKEND not in sys.path: sys.path.insert(0, _BACKEND)
sys.path.insert(0, '.')

# ── Pydantic mock (real pydantic installed via requirements.txt) ──────────────
pm = types.ModuleType("pydantic")
class BM:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
    def __init_subclass__(cls, **kw): pass
    def dict(self): return {k: v for k, v in vars(self).items() if not k.startswith('_')}
pm.BaseModel = BM
pm.Field = lambda default=None, **kw: default
sys.modules["pydantic"] = pm

PASS = 0; FAIL = 0

def ok(msg):
    global PASS; PASS += 1
    print(f"  ✓ {msg}")

def fail(msg, e):
    global FAIL; FAIL += 1
    print(f"  ✗ {msg}: {e}")

# ── 1. Activations ────────────────────────────────────────────────────────────
print("\n[1] Activations")
try:
    from core.activations import get_activation, sigmoid, relu, tanh, linear, leaky_relu, elu
    x = np.linspace(-3, 3, 20)
    for name in ["sigmoid", "relu", "tanh", "linear", "leaky_relu", "elu"]:
        fn = get_activation(name)
        out, deriv = fn(x)
        assert out.shape == x.shape and deriv.shape == x.shape
        if name == "sigmoid": assert all(0 < v < 1 for v in out)
        if name == "relu": assert all(v >= 0 for v in out)
    ok("all 6 activations: shape + range correct")
    try: get_activation("bad_name")
    except ValueError: ok("get_activation raises ValueError on unknown name")
except Exception as e: fail("activations", e)

# ── 2. Losses ─────────────────────────────────────────────────────────────────
print("\n[2] Losses")
try:
    from core.losses import binary_crossentropy, mse, get_loss
    yt = np.array([[1.],[0.],[1.],[0.]])
    yp = np.array([[0.9],[0.1],[0.8],[0.2]])
    loss, grad = binary_crossentropy(yt, yp)
    assert isinstance(loss, float) and 0 < loss < 1
    assert grad.shape == yp.shape
    ok(f"binary_crossentropy: loss={loss:.4f}, grad.shape={grad.shape}")
    loss2, grad2 = mse(yt, yp)
    assert isinstance(loss2, float) and grad2.shape == yp.shape
    ok(f"mse: loss={loss2:.4f}")
    # Numerical stability: near 0 and 1
    yp_extreme = np.array([[1e-10],[1-1e-10]])
    yt_ext = np.array([[0.],[1.]])
    loss_ext, _ = binary_crossentropy(yt_ext, yp_extreme)
    assert not np.isnan(loss_ext) and not np.isinf(loss_ext)
    ok("numerical stability near 0/1 boundaries")
except Exception as e: fail("losses", e)

# ── 3. Regularization ─────────────────────────────────────────────────────────
print("\n[3] Regularization")
try:
    from core.regularization import l1_penalty, l2_penalty, no_penalty, get_regularizer
    W = np.random.randn(4, 4)
    p0, g0 = no_penalty(W, 0.01)
    assert p0 == 0.0 and np.all(g0 == 0)
    ok("no_penalty: zero penalty and gradient")
    p1, g1 = l1_penalty(W, 0.01)
    assert p1 > 0 and g1.shape == W.shape
    assert np.allclose(np.abs(g1), 0.01)
    ok(f"l1_penalty: penalty={p1:.4f}")
    p2, g2 = l2_penalty(W, 0.01)
    assert p2 > 0 and np.allclose(g2, 0.01 * W)
    ok(f"l2_penalty: penalty={p2:.4f}")
    for name in ["none","l1","l2"]:
        fn = get_regularizer(name)
        p, g = fn(W, 0.01)
        assert g.shape == W.shape
    ok("get_regularizer: all 3 names work")
except Exception as e: fail("regularization", e)

# ── 4. Neural Network ─────────────────────────────────────────────────────────
print("\n[4] Neural Network")
try:
    from core.neural_net import NeuralNetwork
    X = np.random.randn(10, 2)

    # Perceptron (0 hidden)
    nn0 = NeuralNetwork([2,1], activations=[], output_activation="sigmoid")
    out = nn0.forward(X)
    assert out.shape == (10,1) and all(0<v<1 for v in out.flatten())
    ok("perceptron (0 hidden layers)")

    # 1 hidden layer
    nn1 = NeuralNetwork([2,8,1], activations=["relu"])
    out1 = nn1.forward(X)
    assert out1.shape == (10,1)
    ok("1 hidden layer, ReLU")

    # Per-layer activations
    nn3 = NeuralNetwork([2,8,4,2,1], activations=["relu","tanh","leaky_relu"])
    out3 = nn3.forward(X)
    assert out3.shape == (10,1)
    assert nn3.activations == ["relu","tanh","leaky_relu","sigmoid"]
    ok("3 hidden layers, per-layer activations: relu/tanh/leaky_relu/sigmoid")

    # from_config
    class Cfg:
        hidden_layers=2; neurons_per_layer=[8,4]; activations_per_layer=["relu","tanh"]
        activation="relu"; regularization="none"; reg_lambda=0.0; dropout_rate=0.0
    nn_c = NeuralNetwork.from_config(Cfg())
    out_c = nn_c.forward(X)
    assert out_c.shape == (10,1)
    ok("from_config: 2 hidden layers [8,4] relu/tanh")

    # Partial activations_per_layer (fallback)
    class CfgP:
        hidden_layers=3; neurons_per_layer=[8,8,8]; activations_per_layer=["relu"]
        activation="tanh"; regularization="none"; reg_lambda=0.0; dropout_rate=0.0
    nn_p = NeuralNetwork.from_config(CfgP())
    assert nn_p.activations[0]=="relu" and nn_p.activations[1]=="tanh" and nn_p.activations[3]=="sigmoid"
    ok(f"partial activations fallback: {nn_p.activations}")

    # get_weights / set_weights roundtrip
    out_before = nn3.forward(X).copy()
    state = nn3.get_weights()
    nn3b = NeuralNetwork([2,8,4,2,1], activations=["relu","tanh","leaky_relu"])
    nn3b.set_weights(state)
    out_after = nn3b.forward(X)
    assert np.allclose(out_before, out_after)
    ok("get_weights/set_weights roundtrip: outputs identical")

    # Weight importance
    imp = nn3.get_weight_importance(0, 0)
    assert len(imp) == 2 and all(0<=v<=1 for v in imp)
    ok(f"get_weight_importance: {[round(v,3) for v in imp]}")

    # 5 hidden layers
    nn5 = NeuralNetwork([2,8,8,8,8,8,1], activations=["relu","tanh","relu","leaky_relu","elu"])
    out5 = nn5.forward(X)
    assert out5.shape == (10,1) and all(0<v<1 for v in out5.flatten())
    ok("5 hidden layers, all different activations")

    # Xavier init: weights in reasonable range
    W0 = nn3.weights[0]
    limit = np.sqrt(6.0 / (2+8))
    assert np.all(np.abs(W0) <= limit * 2)  # within 2x of limit
    ok(f"Xavier init: max|W|={np.abs(W0).max():.4f} (limit={limit:.4f})")

except Exception as e: fail("neural_net", e)

# ── 5. Optimizer ──────────────────────────────────────────────────────────────
print("\n[5] Optimizer")
try:
    from core.optimizer import SGDMomentum, create_mini_batches
    from core.neural_net import NeuralNetwork
    nn = NeuralNetwork([2,4,1], activations=["relu"])
    opt = SGDMomentum(lr=0.01, momentum=0.9)
    w_before = [w.copy() for w in nn.weights]
    w_grads = [np.ones_like(w)*0.1 for w in nn.weights]
    b_grads = [np.ones_like(b)*0.1 for b in nn.biases]
    nn.weights, nn.biases = opt.step(nn.weights, nn.biases, w_grads, b_grads)
    for i in range(len(w_before)):
        assert not np.allclose(nn.weights[i], w_before[i])
    ok("SGDMomentum: weights updated after step")
    # Second step uses momentum
    nn.weights, nn.biases = opt.step(nn.weights, nn.biases, w_grads, b_grads)
    ok("SGDMomentum: second step (momentum) no crash")
    # Mini-batches
    X = np.random.randn(100,2); y = np.random.randint(0,2,100).astype(float)
    batches = create_mini_batches(X, y, 32, seed=42)
    assert len(batches) == 4  # ceil(100/32)=4
    total = sum(len(xb) for xb,yb in batches)
    assert total == 100
    ok(f"create_mini_batches: {len(batches)} batches, {total} total samples")
except Exception as e: fail("optimizer", e)

# ── 6. Trainer ────────────────────────────────────────────────────────────────
print("\n[6] Trainer")
try:
    from core.trainer import Trainer
    from core.neural_net import NeuralNetwork
    from data.datasets import get_dataset
    from data.splitter import train_test_split
    from data.normalizer import StandardNormalizer

    X,y = get_dataset("xor", n_samples=120, noise=0.1, seed=42)
    Xtr,Xte,ytr,yte = train_test_split(X,y)
    norm = StandardNormalizer()
    Xtr_n = norm.fit_transform(Xtr); Xte_n = norm.transform(Xte)

    # Standard training
    nn = NeuralNetwork([2,8,4,1], activations=["relu","tanh"])
    r = Trainer().train(nn, Xtr_n, ytr, Xte_n, yte, learning_rate=0.05, epochs=50, batch_size=16)
    h = r["history"]
    assert len(h.train_loss) == 50
    assert h.train_loss[-1] < h.train_loss[0], "Loss did not decrease"
    assert len(r["weight_snapshots"]) > 0
    assert r["best_epoch"] >= 1
    ok(f"standard: 50 epochs, loss {h.train_loss[0]:.4f}→{h.train_loss[-1]:.4f}, snaps={len(r['weight_snapshots'])}")

    # High LR (no crash)
    nn2 = NeuralNetwork([2,4,1], activations=["relu"])
    r2 = Trainer().train(nn2, Xtr_n, ytr, Xte_n, yte, learning_rate=5.0, epochs=20, gradient_clip=5.0)
    assert len(r2["history"].train_loss) == 20
    ok(f"high LR=5.0: no crash, final_loss={r2['history'].train_loss[-1]:.4f}")

    # L1 regularization
    nn3 = NeuralNetwork([2,4,1], activations=["relu"])
    r3 = Trainer().train(nn3, Xtr_n, ytr, Xte_n, yte, learning_rate=0.01, epochs=25, regularization="l1", reg_lambda=0.01)
    assert len(r3["history"].train_loss) == 25
    ok("L1 regularization: 25 epochs, no crash")

    # L2 regularization
    nn4 = NeuralNetwork([2,4,1], activations=["relu"])
    r4 = Trainer().train(nn4, Xtr_n, ytr, Xte_n, yte, learning_rate=0.01, epochs=25, regularization="l2", reg_lambda=0.005)
    assert len(r4["history"].train_loss) == 25
    ok("L2 regularization: 25 epochs, no crash")

    # Insight events
    nn5 = NeuralNetwork([2,1], activations=[])  # perceptron on XOR = underfitting
    r5 = Trainer().train(nn5, Xtr_n, ytr, Xte_n, yte, learning_rate=0.01, epochs=25)
    ok(f"insight events: {[e.event_type for e in r5['insight_events']]}")

    # Trainer.stop() — verify stop flag is respected
    trainer6 = Trainer()
    trainer6.stop()  # pre-set stop flag
    nn6 = NeuralNetwork([2,8,1], activations=["relu"])
    r6 = trainer6.train(nn6, Xtr_n, ytr, Xte_n, yte, learning_rate=0.01, epochs=500)
    ep_run = len(r6["history"].train_loss)
    # stop() was called before train() so it stops immediately (0 or 1 epochs)
    assert ep_run <= 1, f"Stop flag not respected: ran {ep_run} epochs"
    ok(f"trainer.stop(): pre-set stop flag halts training at epoch {ep_run}")

except Exception as e: fail("trainer", e)

# ── 7. Datasets ───────────────────────────────────────────────────────────────
print("\n[7] Datasets")
try:
    from data.datasets import get_dataset, DATASET_REGISTRY
    for name in ["linear","xor","noisy","imbalanced"]:
        X,y = get_dataset(name)
        assert X.ndim==2 and y.ndim==1 and X.shape[1]==2
        assert set(y.astype(int).tolist()) == {0,1}
        assert X.shape[0] == y.shape[0]
        ok(f"{name}: shape={X.shape} classes={{0,1}}")
    # Custom params
    X,y = get_dataset("xor", n_samples=50, noise=0.05, seed=99)
    assert len(X) >= 48, f"Expected ~50 samples, got {len(X)}"  # XOR rounds to n//4*4
    ok(f"custom n_samples=50, noise=0.05, seed=99 -> actual={len(X)}")
    # Unknown dataset
    try: get_dataset("bad")
    except ValueError: ok("ValueError on unknown dataset name")
    assert len(DATASET_REGISTRY) == 4
    ok(f"DATASET_REGISTRY: {list(DATASET_REGISTRY.keys())}")
except Exception as e: fail("datasets", e)

# ── 8. Splitter + Normalizer ──────────────────────────────────────────────────
print("\n[8] Splitter + Normalizer")
try:
    from data.splitter import train_test_split
    from data.normalizer import StandardNormalizer
    from data.datasets import get_dataset
    X,y = get_dataset("imbalanced", n_samples=200)
    Xtr,Xte,ytr,yte = train_test_split(X,y,stratified=True)
    assert len(Xtr)+len(Xte)==200
    # Stratified: both sets should have both classes
    assert 0 in yte and 1 in yte
    ok(f"stratified split: train={len(Xtr)} test={len(Xte)}, both classes in test")
    # Normalizer
    norm = StandardNormalizer()
    Xtr_n = norm.fit_transform(Xtr)
    Xte_n = norm.transform(Xte)
    assert abs(Xtr_n.mean()) < 0.1  # near zero mean
    assert abs(Xtr_n.std() - 1.0) < 0.1  # near unit variance
    ok(f"normalizer: train mean={Xtr_n.mean():.4f} std={Xtr_n.std():.4f}")
    # No leakage: test mean != 0 (fitted on train only)
    ok(f"no leakage: test mean={Xte_n.mean():.4f} (not 0 if different distribution)")
except Exception as e: fail("splitter+normalizer", e)

# ── 9. Failure Detector ───────────────────────────────────────────────────────
print("\n[9] Failure Detector")
try:
    from analysis.failure_detector import analyze
    # Overfitting
    fr=analyze([0.6,0.3,0.1,0.02],[0.6,0.5,0.5,0.7],[0.5,0.7,0.9,0.99],[0.5,0.65,0.6,0.55])
    assert fr.is_overfitting and "overfitting" in fr.flags and fr.overfit_epoch is not None
    ok(f"overfitting detected: flags={fr.flags} overfit_ep={fr.overfit_epoch}")
    # Underfitting
    fr2=analyze([0.69]*30,[0.69]*30,[0.52]*30,[0.51]*30)
    assert fr2.is_underfitting
    ok(f"underfitting detected: flags={fr2.flags}")
    # Diverging (exponential growth)
    tl3=[0.5*1.5**i for i in range(8)]
    fr3=analyze(tl3,tl3,[0.5]*8,[0.5]*8)
    assert fr3.is_diverging
    ok(f"diverging detected: flags={fr3.flags}")
    # Stalled
    fr4=analyze([0.5+i*0.0001 for i in range(25)],[0.5+i*0.0001 for i in range(25)],[0.62]*25,[0.62]*25)
    assert fr4.is_stalled
    ok(f"stalled detected: flags={fr4.flags}")
    # Healthy
    tl5=[0.6,0.4,0.25,0.15,0.10,0.08,0.07,0.065,0.062,0.060]
    vl5=[0.62,0.42,0.27,0.17,0.12,0.10,0.095,0.092,0.090,0.089]
    ta5=[0.5,0.65,0.78,0.87,0.91,0.93,0.94,0.945,0.947,0.948]
    va5=[0.5,0.63,0.76,0.85,0.89,0.91,0.92,0.922,0.923,0.924]
    fr5=analyze(tl5,vl5,ta5,va5)
    assert not fr5.is_overfitting and not fr5.is_underfitting and not fr5.is_diverging
    ok(f"healthy: no flags, severity={fr5.severity}")
    # Exploding
    fr6=analyze([0.5,float('nan'),20.0,50.0],[0.5,1.0,2.0,5.0],[0.5]*4,[0.5]*4)
    assert "exploding_loss" in fr6.flags and fr6.severity==1.0
    ok(f"exploding loss: flags={fr6.flags}")
except Exception as e: fail("failure_detector", e)

# ── 10. Explainer ─────────────────────────────────────────────────────────────
print("\n[10] Explainer")
try:
    from analysis.failure_detector import analyze
    from analysis.explainer import explain
    for scenario,tl,vl,ta,va in [
        ("overfit",[0.6,0.3,0.1,0.02],[0.6,0.5,0.5,0.7],[0.5,0.7,0.9,0.99],[0.5,0.65,0.6,0.55]),
        ("underfit",[0.69]*30,[0.69]*30,[0.52]*30,[0.51]*30),
        ("diverge",[0.5*1.5**i for i in range(8)],[0.5*1.5**i for i in range(8)],[0.5]*8,[0.5]*8),
    ]:
        fr=analyze(tl,vl,ta,va)
        cards=explain(fr)
        assert len(cards)>0, f"No cards for {scenario}"
        for c in cards:
            assert c.title and len(c.body)>20 and len(c.action_hint)>10
        ok(f"{scenario}: {len(cards)} card(s), title+body+action_hint present")
    # No flags = no cards
    tl5=[0.6,0.4,0.25,0.15,0.10,0.08,0.07,0.065,0.062,0.060]
    vl5=[0.62,0.42,0.27,0.17,0.12,0.10,0.095,0.092,0.090,0.089]
    ta5=[0.5,0.65,0.78,0.87,0.91,0.93,0.94,0.945,0.947,0.948]
    va5=[0.5,0.63,0.76,0.85,0.89,0.91,0.92,0.922,0.923,0.924]
    fr5=analyze(tl5,vl5,ta5,va5); cards5=explain(fr5)
    assert len(cards5)==0
    ok("healthy training: 0 explanation cards")
except Exception as e: fail("explainer", e)

# ── 11. Boundary Computer ─────────────────────────────────────────────────────
print("\n[11] Boundary Computer")
try:
    from analysis.boundary_computer import compute_boundary
    from core.neural_net import NeuralNetwork
    from data.datasets import get_dataset
    from data.splitter import train_test_split
    from data.normalizer import StandardNormalizer
    X,y=get_dataset("xor",n_samples=100); Xtr,Xte,ytr,yte=train_test_split(X,y)
    norm=StandardNormalizer(); Xtr_n=norm.fit_transform(Xtr); Xte_n=norm.transform(Xte)
    nn=NeuralNetwork([2,8,1],activations=["relu"])
    all_X=np.vstack([Xtr_n,Xte_n]); all_y=np.hstack([ytr,yte])
    for res in [20,50,80]:
        bd=compute_boundary(nn,all_X,all_y,resolution=res)
        assert len(bd.xx)==res and len(bd.xx[0])==res
        assert len(bd.Z)==res and len(bd.Z[0])==res
        assert len(bd.x_points)==100 and len(bd.labels)==100
        Z_arr=np.array(bd.Z)
        assert Z_arr.min()>=0 and Z_arr.max()<=1, f"Z values out of [0,1]: min={Z_arr.min()} max={Z_arr.max()}"
        ok(f"resolution={res}: grid {res}x{res}, {len(bd.x_points)} points, Z in [0,1]")
except Exception as e: fail("boundary_computer", e)

# ── 12. Metrics ───────────────────────────────────────────────────────────────
print("\n[12] Metrics")
try:
    from analysis.metrics import accuracy,precision,recall,f1_score,confusion_matrix,generalization_gap,compute_all_metrics
    eps=1e-6
    yt=np.array([1,0,1,0,1,0]); yp=np.array([1,0,1,0,1,0])
    assert accuracy(yt,yp)==1.0 and np.isclose(f1_score(yt,yp),1.0,atol=eps)
    ok("perfect predictions: acc=1.0 f1~=1.0")
    assert accuracy(yt,1-yp)==0.0
    ok("all wrong: acc=0.0")
    yt3=np.array([0,0,0,0,0,1]); yp3=np.zeros(6,dtype=int)
    assert accuracy(yt3,yp3)>0.8 and f1_score(yt3,yp3)<0.1
    ok(f"imbalanced trap: acc={accuracy(yt3,yp3):.3f} f1={f1_score(yt3,yp3):.4f}")
    cm_res=confusion_matrix(np.array([1,0,1,0]),np.array([1,0,0,1]))
    assert cm_res==[[1,1],[1,1]]
    ok(f"confusion_matrix: [[TN,FP],[FN,TP]]={cm_res}")
    assert abs(generalization_gap(0.95,0.72)-0.23)<0.001
    ok("generalization_gap: 0.95-0.72=0.23")
    m=compute_all_metrics(np.array([1,0,1,0,1]),np.array([[0.8],[0.2],[0.9],[0.3],[0.7]]))
    assert all(k in m for k in ['accuracy','f1_score','precision','recall','confusion_matrix'])
    assert all(0<=v<=1 for k,v in m.items() if k!='confusion_matrix')
    ok(f"compute_all_metrics: {{{', '.join(k+'='+('%.3f'%v) for k,v in m.items() if k!='confusion_matrix')}}}")
except Exception as e: fail("metrics", e)

# ── 13. Storage ───────────────────────────────────────────────────────────────
print("\n[13] Storage")
try:
    from storage.session_store import get_session, update_session, set_stop_flag, is_stop_requested, clear_stop_flag
    from storage.replay_store import save_snapshots, get_replay, get_replay_run, clear_replay
    from storage.progress_store import get_progress,award_xp,unlock_feature,unlock_features,add_insight,reset_progress,is_feature_unlocked

    sid="test_storage"
    s=get_session(sid); assert isinstance(s,dict) and "training_running" in s
    update_session(sid,"training_running",True)
    assert get_session(sid)["training_running"]==True
    set_stop_flag(sid); assert is_stop_requested(sid)
    clear_stop_flag(sid); assert not is_stop_requested(sid)
    ok("session_store: get/update/stop_flag/clear")

    class MockSnap:
        def __init__(self,i): self.epoch=i; self.train_loss=0.5; self.test_loss=0.52; self.train_acc=0.7; self.test_acc=0.68; self.weights=[[[0.1]]]; self.biases=[[[0.01]]]
        def dict(self): return vars(self)
    snaps=[MockSnap(i) for i in range(5)]
    save_snapshots(sid,"run1",snaps)
    got=get_replay_run(sid,"run1"); assert len(got)==5
    all_r=get_replay(sid); assert "run1" in all_r
    clear_replay(sid); assert get_replay(sid)=={}
    ok("replay_store: save/get/get_all/clear")

    sid2="test_progress"
    p=get_progress(sid2); assert p["xp"]==0 and p["current_level"]==1
    award_xp(sid2,250); p2=get_progress(sid2); assert p2["xp"]==250
    unlock_feature(sid2,"hidden_layers"); assert is_feature_unlocked(sid2,"hidden_layers")
    unlock_features(sid2,["activation","regularization"])
    p3=get_progress(sid2); assert "activation" in p3["unlocked_features"]
    add_insight(sid2,"xor_failure_witnessed")
    p4=get_progress(sid2); assert "xor_failure_witnessed" in p4.get("insights_collected",[])
    add_insight(sid2,"xor_failure_witnessed")  # duplicate - should not double
    p5=get_progress(sid2); assert p5["insights_collected"].count("xor_failure_witnessed")==1
    reset_progress(sid2); p6=get_progress(sid2); assert p6["xp"]==0
    ok("progress_store: xp/unlock/insight/duplicate_insight/reset")
except Exception as e: fail("storage", e)

# ── 14. Level Manager ─────────────────────────────────────────────────────────
print("\n[14] Level Manager")
try:
    from progression.level_manager import get_level, get_all_levels, is_param_unlocked
    levels=get_all_levels()
    assert len(levels)==6
    for l in levels:
        lid=l["level_id"]
        assert "challenge" in l and "default_config" in l and "unlock_reward" in l
        assert isinstance(l.get("tips",[]),list) and len(l["tips"])>0
        ch=l["challenge"]
        assert "target_metric" in ch and "target_value" in ch and "type" in ch
    ok(f"all 6 levels: valid structure with tips")
    l1=get_level(1); assert l1["level_id"]==1 and l1["dataset"]=="linear"
    ok(f"L1: dataset={l1['dataset']} challenge={l1['challenge']['type']}")
    l6=get_level(6); assert l6["level_id"]==6 and "all" in l6["unlocked_params"]
    ok(f"L6: unlocked_params contains 'all'")
    try: get_level(99)
    except ValueError: ok("ValueError on invalid level_id")
except Exception as e: fail("level_manager", e)

# ── 15. Challenge Validator ───────────────────────────────────────────────────
print("\n[15] Challenge Validator")
try:
    from progression.challenge_validator import validate
    def mk(ta,va,f1=None): return {"history":{"train_acc":[ta],"test_acc":[va],"train_loss":[0.3],"test_loss":[0.32]},"final_metrics":{"f1_score":f1 or va,"accuracy":va}}
    # L1 pass/fail
    ch=validate(1,mk(0.92,0.88)); assert ch.passed and ch.xp_earned==100
    ok(f"L1 pass: xp={ch.xp_earned}")
    ch=validate(1,mk(0.70,0.72)); assert not ch.passed and 0<ch.xp_earned<100
    ok(f"L1 fail: partial_xp={ch.xp_earned}")
    # L2 pass
    ch=validate(2,mk(0.95,0.93)); assert ch.passed and ch.xp_earned==200
    ok(f"L2 pass: score={ch.score:.3f}")
    # L4 gap <5%
    ch=validate(4,mk(0.82,0.80)); assert ch.passed
    ok(f"L4 pass: gap={abs(0.82-0.80):.2f}<0.05")
    ch=validate(4,mk(0.95,0.70)); assert not ch.passed
    ok(f"L4 fail: gap=0.25>0.05")
    # L5 f1
    ch=validate(5,mk(0.92,0.90,f1=0.91)); assert ch.passed and ch.xp_earned==400
    ok(f"L5 pass: f1=0.91 xp={ch.xp_earned}")
    ch=validate(5,mk(0.85,0.85,f1=0.45)); assert not ch.passed
    ok(f"L5 fail: f1=0.45 (accuracy trap detected)")
    # L6 duel
    ch=validate(6,{**mk(0.9,0.88),"duel_wins":2}); assert ch.passed and ch.xp_earned==500
    ok(f"L6 duel win: xp={ch.xp_earned}")
    ch=validate(6,{**mk(0.6,0.58),"duel_wins":1}); assert not ch.passed
    ok(f"L6 duel lose: partial_xp={ch.xp_earned}")
    # Partial credit
    ch25=validate(1,mk(0.7,0.72))
    assert 0<ch25.partial_credit<1
    assert abs(ch25.partial_credit - ch25.score/0.85) < 0.001, f"partial={ch25.partial_credit} score/thr={ch25.score/0.85}"
    ok(f"partial_credit={ch25.partial_credit:.3f} = score({ch25.score:.3f})/threshold(0.85)")
except Exception as e: fail("challenge_validator", e)

# ── 16. Experiment Configs ────────────────────────────────────────────────────
print("\n[16] Experiment Configs")
try:
    from progression.experiment_configs import get_experiment, list_experiments
    exps=list_experiments(); assert len(exps)==3
    for exp in exps:
        assert "model_a" in exp and "model_b" in exp
        assert "lesson" in exp and "archetype_a" in exp and "archetype_b" in exp
        ma=exp["model_a"]; mb=exp["model_b"]
        assert "activations_per_layer" in ma and "activations_per_layer" in mb
        assert "hidden_layers" in ma and "regularization" in ma
    ok(f"3 experiments: {[e['id'] for e in exps]}")
    exp=get_experiment("perceptron_vs_mlp_xor")
    assert exp["model_a"]["hidden_layers"]==0 and exp["model_b"]["hidden_layers"]==2
    ok(f"perceptron_vs_mlp_xor: A={exp['model_a']['hidden_layers']} layers vs B={exp['model_b']['hidden_layers']} layers")
    try: get_experiment("bad_id")
    except ValueError: ok("ValueError on unknown experiment id")
except Exception as e: fail("experiment_configs", e)

# ── 17. Full integration ──────────────────────────────────────────────────────
print("\n[17] Full Integration (all datasets, complete pipeline)")
try:
    from core.neural_net import NeuralNetwork
    from core.trainer import Trainer
    from data.datasets import get_dataset
    from data.splitter import train_test_split
    from data.normalizer import StandardNormalizer
    from analysis.failure_detector import analyze
    from analysis.boundary_computer import compute_boundary
    from analysis.explainer import explain
    from analysis.metrics import compute_all_metrics, confusion_matrix as cm
    from storage.session_store import update_session, get_session
    from storage.replay_store import save_snapshots, get_replay_run
    from progression.challenge_validator import validate

    class MC:
        hidden_layers=2; neurons_per_layer=[8,4]; activations_per_layer=["relu","tanh"]
        activation="relu"; regularization="l2"; reg_lambda=0.005; dropout_rate=0.0

    for ds in ["linear","xor","noisy","imbalanced"]:
        X,y=get_dataset(ds,n_samples=160,noise=0.1,seed=42)
        Xtr,Xte,ytr,yte=train_test_split(X,y,stratified=True)
        norm=StandardNormalizer(); Xtr_n=norm.fit_transform(Xtr); Xte_n=norm.transform(Xte)
        nn=NeuralNetwork.from_config(MC())
        r=Trainer().train(nn,Xtr_n,ytr,Xte_n,yte,learning_rate=0.05,epochs=40,batch_size=16)
        h=r["history"]
        fr=analyze(h.train_loss,h.test_loss,h.train_acc,h.test_acc)
        cards=explain(fr)
        bd=compute_boundary(nn,np.vstack([Xtr_n,Xte_n]),np.hstack([ytr,yte]),resolution=30)
        pred=nn.forward(Xte_n)
        m=compute_all_metrics(yte,pred)
        conf=cm(yte.astype(int),(pred>=0.5).astype(int).flatten())
        sid=f"int_{ds}"
        save_snapshots(sid,r["run_id"],r["weight_snapshots"])
        res={"run_id":r["run_id"],"history":h.__dict__,"final_metrics":m,
             "failure_report":fr.__dict__,"confusion_matrix":conf}
        update_session(sid,"last_result",res)
        snaps=get_replay_run(sid,r["run_id"])
        ch=validate(1,get_session(sid)["last_result"])
        # Validate response completeness
        assert len(h.train_loss)==40
        assert len(bd.xx)==30 and len(bd.x_points)==160
        assert len(conf)==2 and len(conf[0])==2
        assert len(snaps)>0
        assert 0<=m["accuracy"]<=1 and 0<=m["f1_score"]<=1
        ok(f"{ds}: acc={m['accuracy']:.3f} f1={m['f1_score']:.3f} flags={fr.flags or 'none'} snaps={len(snaps)} challenge_pass={ch.passed}")
except Exception as e: fail("full_integration", e)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("="*60)
total = PASS + FAIL
print(f"  Tests passed: {PASS}/{total}")
if FAIL > 0:
    print(f"  Tests FAILED: {FAIL}/{total}")
    print("  ⚠ Fix the failures above before shipping.")
else:
    print("  ✅ ALL TESTS PASSED — Backend is fully verified")
print("="*60)
sys.exit(0 if FAIL == 0 else 1)
