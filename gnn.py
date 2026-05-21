import os
import h5py
import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
import optax
from flax.training import train_state
from flax import serialization
from typing import Sequence, Callable

# ---------------------------------------------------------
# 1. MLP Modules (Replacing `modules as mod`)
# ---------------------------------------------------------
class MLP(nn.Module):
    """Standard Multi-Layer Perceptron used for node, edge, and global updates."""
    layer_sizes: Sequence[int]
    activation: Callable = nn.relu
    output_activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes[:-1]:
            x = nn.Dense(size)(x)
            x = self.activation(x)
        # Final layer uses the specific output activation
        x = nn.Dense(self.layer_sizes[-1])(x)
        x = self.output_activation(x)
        return x

# ---------------------------------------------------------
# 2. Graph Layer Blocks
# ---------------------------------------------------------
def make_update_edge_fn(dim_out, n_layers, out_act):
    @jraph.concatenated_args
    def update_edge_fn(features):
        return MLP([dim_out] * n_layers, output_activation=out_act)(features)
    return update_edge_fn

def make_update_node_fn(dim_out, n_layers, out_act):
    @jraph.concatenated_args
    def update_node_fn(features):
        return MLP([dim_out] * n_layers, output_activation=out_act)(features)
    return update_node_fn

def make_update_global_fn(dim_out, n_layers, out_act):
    @jraph.concatenated_args
    def update_global_fn(features):
        return MLP([dim_out] * n_layers, output_activation=out_act)(features)
    return update_global_fn

# ---------------------------------------------------------
# 3. WPGNN Architecture (Stateless Flax Module)
# ---------------------------------------------------------
class WPGNN(nn.Module):
    """
    WPGNN Architecture in Flax.
    Note: Flax modules are stateless. They only define the computation graph.
    """
    graph_sizes: Sequence[Sequence[int]] = (
        (32, 32, 32),
        (16, 16, 16),
        (16, 16, 16),
        (8, 8, 8),
        (8, 8, 8),
        (4, 2, 2)
    )

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        n_layers = len(self.graph_sizes)

        for i, dim_out in enumerate(self.graph_sizes):
            is_last_layer = (i == n_layers - 1)
            out_act = nn.relu if is_last_layer else nn.sigmoid
            n_mlp_layers = 1 if is_last_layer else 2

            e_out, n_out, g_out = dim_out

            gn = jraph.GraphNetwork(
                update_edge_fn=make_update_edge_fn(e_out, n_mlp_layers, out_act),
                update_node_fn=make_update_node_fn(n_out, n_mlp_layers, out_act),
                update_global_fn=make_update_global_fn(g_out, n_mlp_layers, out_act)
            )

            graph_out = gn(graph)

            # Residual connections (if dimensions match)
            if graph.edges.shape[-1] == graph_out.edges.shape[-1] and \
                    graph.nodes.shape[-1] == graph_out.nodes.shape[-1] and \
                    graph.globals.shape[-1] == graph_out.globals.shape[-1]:
                graph_out = graph_out._replace(
                    edges=graph.edges + graph_out.edges,
                    nodes=graph.nodes + graph_out.nodes,
                    globals=graph.globals + graph_out.globals
                )

            graph = graph_out

        return graph

# ---------------------------------------------------------
# 4. Training Manager (Holding State, Loss, and Fit routines)
# ---------------------------------------------------------
class WPGNNTrainer:
    def __init__(self, eN=2, nN=3, gN=3, graph_size=None, scale_factors=None, model_path=None, learning_rate=1e-3):
        self.model = WPGNN(graph_sizes=graph_size if graph_size else WPGNN.graph_sizes)

        # Scale factors
        self.scale_factors = scale_factors or {
            'x_globals': jnp.array([[0., 25.], [0., 25.], [0.09, 0.03]]),
            'x_nodes': jnp.array([[0., 75000.], [0., 85000.], [15., 15.]]),
            'x_edges': jnp.array([[-100000., 100000.], [0., 75000.]]),
            'f_globals': jnp.array([[0., 500000000.], [0., 100000.]]),
            'f_nodes': jnp.array([[0., 5000000.], [0., 25.]]),
            'f_edges': jnp.array([[0., 0.]])
        }

        # Initialize network weights with dummy data
        dummy_graph = jraph.GraphsTuple(
            n_node=jnp.array([2]),
            n_edge=jnp.array([2]),
            nodes=jnp.ones((2, nN)),
            edges=jnp.ones((2, eN)),
            globals=jnp.ones((1, gN)),
            senders=jnp.array([1, 0]),
            receivers=jnp.array([0, 1])
        )
        key = jax.random.PRNGKey(0)
        variables = self.model.init(key, dummy_graph)

        # Setup Optimizer with exponential decay
        self.tx = optax.adam(learning_rate)

        # Create Train State
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=self.tx,
        )

        if model_path is not None:
            self.load_weights(model_path)

    def compute_loss(self, params, x: jraph.GraphsTuple, f: jraph.GraphsTuple):
        """Pure function for loss computation to be utilized by JAX transformations."""
        x_out = self.model.apply({'params': params}, x)

        # MSE for turbine (nodes) and plant (globals)
        turbine_loss = jnp.mean((x_out.nodes - f.nodes)**2, axis=0)
        plant_loss = jnp.mean((x_out.globals - f.globals)**2, axis=0)

        # Combined weighted loss
        loss = jnp.sum(plant_loss) + 10. * jnp.sum(turbine_loss)

        return loss, (turbine_loss, plant_loss)

    @jax.jit
    def train_step(self, state, x: jraph.GraphsTuple, f: jraph.GraphsTuple):
        """Jitted training step computing gradients and applying them."""
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, aux), grads = grad_fn(state.params, x, f)

        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    @jax.jit
    def eval_step(self, state, x: jraph.GraphsTuple, f: jraph.GraphsTuple):
        """Jitted evaluation step."""
        loss, aux = self.compute_loss(state.params, x, f)
        return loss, aux

    def fit(self, train_data, test_data=None, batch_size=100, epochs=100,
            print_every=10, save_every=100, save_model_path=None):

        # Train data should ideally be formatted into lists of jraph.GraphsTuple
        # that are pre-batched using jraph.batch()

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            print(f'Beginning epoch {epoch}...')

            for iters, (x_batch, f_batch) in enumerate(train_data):
                self.state, loss, (t_loss, p_loss) = self.train_step(self.state, x_batch, f_batch)

                if print_every > 0 and (iters % print_every) == 0:
                    print(f'Total batch loss = {loss:.6f}')
                    print(f'Turbine power loss = {t_loss[0]:.6f}, turbine speed loss = {t_loss[1]:.6f}')
                    print(f'Plant power loss   = {p_loss[0]:.6f}, plant cabling loss = {p_loss[1]:.6f}\n')

            # Save state
            if save_model_path and (epoch % save_every) == 0:
                epoch_dir = os.path.join(save_model_path, f'{epoch:05d}')
                os.makedirs(epoch_dir, exist_ok=True)
                self.save_weights(os.path.join(epoch_dir, 'wpgnn_flax.msgpack'))

            print(f'Epochs {epoch} Complete. Time: {time.time() - start_time:.2f}s\n')

    # ---------------------------------------------------------
    # I/O Handlers (Replaces custom h5py logic with Flax standard)
    # ---------------------------------------------------------
    def save_weights(self, filename):
        """Saves weights using Flax's built in MessagePack serializer (much safer than custom h5py)."""
        bytes_output = serialization.to_bytes(self.state.params)
        with open(filename, 'wb') as f:
            f.write(bytes_output)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            bytes_input = f.read()
        new_params = serialization.from_bytes(self.state.params, bytes_input)
        self.state = self.state.replace(params=new_params)