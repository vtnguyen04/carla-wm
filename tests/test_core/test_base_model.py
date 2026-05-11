import pytest
import torch
import torch.nn as nn
import os
from torch_wm.core.base_model import Model
from torch_wm.optimizers import Adam
from torch_wm.structs import AttrDict

class MockModel(Model):
    def __init__(self):
        super().__init__(name="MockModel")
        self.net = nn.Linear(10, 2)
        self.built = True
        
    def forward(self, x):
        return self.net(x)

def test_model_compile_weights():
    model = MockModel()
    # Test float weight
    model.compile(losses=nn.MSELoss(), loss_weights=0.5)
    assert model.compiled
    
    # Test list weight
    model.compile(losses=[nn.MSELoss()], loss_weights=[0.1])
    
    # Test dict weight
    model.compile(losses={"out": nn.MSELoss()}, loss_weights={"out": 0.2})

def test_map_to_outputs():
    model = MockModel()
    outputs = {"a": 1, "b": 2}
    
    # Test mapping single item
    mapped = model.map_to_outputs(outputs, 0.5)
    assert mapped == {"a": 0.5, "b": 0.5}
    
    # Test mapping list
    mapped = model.map_to_outputs(outputs, [0.1, 0.2])
    assert mapped == {"a": 0.1, "b": 0.2}
    
    # Test mapping dict with missing key
    mapped = model.map_to_outputs(outputs, {"a": 0.3})
    assert mapped["a"] == 0.3
    assert mapped["b"] is None
    
    # Test mapping dict with unexpected key
    with pytest.raises(Exception):
        model.map_to_outputs(outputs, {"c": 0.4})

def test_model_save_load(tmp_path):
    model = MockModel()
    model.compile(losses=None, optimizer=Adam(model.parameters(), lr=1e-3))
    
    path = tmp_path / "model.ckpt"
    model.save(str(path))
    assert os.path.exists(path)
    
    new_model = MockModel()
    new_model.compile(losses=None, optimizer=Adam(new_model.parameters(), lr=1e-3))
    new_model.load(str(path))
    assert torch.equal(model.net.weight, new_model.net.weight)

def test_model_summary(capsys):
    model = MockModel()
    model.summary(show_dict=True, show_modules=True)
    captured = capsys.readouterr()
    assert "Model name: MockModel" in captured.out
    assert "State Dict:" in captured.out
    assert "Named Modules:" in captured.out

def test_num_params():
    model = MockModel()
    # 10*2 weights + 2 biases = 22 params
    assert model.num_params() == 22
    assert model.num_params(model.net) == 22
    assert model.num_params([model.net]) == 22
