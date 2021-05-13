import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from scipy.integrate import odeint
from scipy import interpolate
import pickle
import multiprocessing as mp
import os
import CFD.cfd


class Actor(nn.Module):

    # Initializing the NN for the actor

    def __init__(self, in_dim: int, out_dim: int, ):
        self.in_dim = in_dim
        self.out_dim = out_dim

        super(Actor, self).__init__()

        self.hidden_one = nn.Linear(in_dim, 350)
        self.hidden_two = nn.Linear(350, 350)
        self.mu_layer = nn.Linear(350, out_dim)
        self.log_std_layer = nn.Linear(350, out_dim)

    # Defining the forward pass for the actor NN

    def forward(self, state: torch.Tensor):
        x = torch.tanh(self.hidden_one(state))
        x = torch.tanh(self.hidden_two(x))

        mu = torch.tanh(self.mu_layer(x))
        log_std = torch.tanh(self.log_std_layer(x))

        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist, mu, std


# Critic network class created below

class Critic(nn.Module):

    # Initializing the NN for the critic
    def __init__(self, in_dim: int):
        super(Critic, self).__init__()

        self.hidden_one = nn.Linear(in_dim, 350)
        self.hidden_two = nn.Linear(350, 350)
        self.out = nn.Linear(350, 1)

    # Defining the forward pass for the critic NN
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden_one(state))
        x = F.relu(self.hidden_two(x))
        value = self.out(x)

        return value


# Class that defines the necessary properties required for the PPO agent

class PPO_Agent(object):

    # We basically pass and initialize all the parameters here
    
    def __init__(self, obs_dim=6, act_dim=12, gamma=0.99, lamda=0.10,
                 entropy_coef=0.001, epsilon=0.25, num_epochs=10, batch_size=250, 
                 actor_lr=1e-4, critic_lr=3.5e-4):

        # Hyperparameters for the RL 
        
        self.gamma = gamma
        self.lamda = lamda
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        #Creates the actor and critic NN 
        
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.critic = Critic(self.obs_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Stores all the data from the environment in a policy
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []
        self.next_states = []
        self.actor_losses = []
        self.critic_losses = []

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Reset the memory after every policy update
    
    def clear_memory(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []
        self.next_states = []
        
    # Calculates the advantage of each action using the value function

    def get_gae(self, rewards: list, values: list, is_terminals: list, gamma: float, lamda: float, ):

        gae = 0
        returns = []

        for i in reversed(range(len(rewards))):
            delta = (rewards[i] + gamma * values[i + 1] * is_terminals[i] - values[i])
            gae = delta + gamma * lamda * is_terminals[i] * gae
            return_ = np.array([gae + values[i]])
            returns.insert(0, return_)

        return returns

    # Selects random data from the memory to use in the policy update. If batch size equals the 
    # memory size (num actions * num episodes) then it will use all the data together for an update. 
    
    def trajectories_data_generator(self, states: torch.Tensor, actions: torch.Tensor,
                                    returns: torch.Tensor, log_probs: torch.Tensor,
                                    values: torch.Tensor, advantages: torch.Tensor,
                                    batch_size, num_epochs, ):
        data_len = states.size(0)
        for _ in range(num_epochs):
            for _ in range(data_len // batch_size):
                ids = np.random.choice(data_len, batch_size)
                yield states[ids, :], actions[ids], returns[ids], log_probs[ids], values[ids], advantages[ids]

    # Based on the state, we get the action from actor NN
    # The actor NN outputs the probability distribution for every state from which an action is sampled from
    
    def _get_action(self, state):

        state = torch.FloatTensor(state).to(self.device)
        action, dist, mu, std = self.actor.forward(state)
        value = self.critic.forward(state)

        return action, value, dist, mu, std

    # We update the weights of the actor and critic NN
    
    def _update_weights(self, lr_manual_bool, actor_lr, critic_lr):

        self.rewards = torch.tensor(self.rewards).float()
        self.is_terminals = torch.tensor(self.is_terminals).float()
        self.values = torch.tensor(self.values).float()
        self.states = torch.tensor(self.states).float()
        self.log_probs = torch.tensor(self.log_probs).float()
        self.actions = torch.tensor(self.actions).float()

        returns = self.get_gae(self.rewards, self.values, self.is_terminals,
                               self.gamma, self.lamda, )
        returns = torch.tensor(returns).float()

        states = self.states
        actions = self.actions
        log_probs = self.log_probs
        values = self.values
        advantages = returns - values[:-1]

        actor_losses = []
        critic_losses = []

        for state, action, return_, old_log_prob, old_value, advantage in self.trajectories_data_generator(
                states=states, actions=actions, returns=returns, log_probs=log_probs, values=values,
                advantages=advantages, batch_size=self.batch_size, num_epochs=self.num_epochs, ):
            
            # compute ratio (pi_theta / pi_theta__old)
            _, dist, __, ___ = self.actor(state)
            cur_log_prob = dist.log_prob(action)
            ratio = torch.exp(cur_log_prob - old_log_prob)

            # compute entropy
            entropy = dist.entropy().mean()

            # compute actor loss
            loss = advantage * ratio
            clipped_loss = (torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon)
                            * advantage)
            actor_loss = (-torch.mean(torch.min(loss, clipped_loss))
                          - entropy * self.entropy_coef)

            # compute critic loss
            cur_value = self.critic(state)
            critic_loss = (return_ - cur_value).pow(2).mean()

            if lr_manual_bool:
                self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

            # actor optimizer step
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # critic optimizer step
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)

    # We save the weights into actor and critic files here for post-processing or other fxns
    def _save_weights(self, policy_num: int):

        filename_actor = "Actor_Policy_Number_" + str(policy_num)
        filename_critic = "Critic_Policy_Number_" + str(policy_num)
        torch.save(self.actor.state_dict(), filename_actor)
        torch.save(self.critic.state_dict(), filename_critic)

    # If necessary, we load the saved weights for a given actor, critic policy here
    def _load_weights(self, filename_actor, filename_critic):

        self.actor.load_state_dict(torch.load(filename_actor))
        self.actor.eval()
        self.critic.load_state_dict(torch.load(filename_critic))
        self.actor.eval()

# This is the PI control
# Motor data generated from the RL is passed through this before being passed to the CFD

def PI_Motor(w_des_two, tsteps, dt, w0):
    w_des_one = np.zeros(50)
    for i in range(len(w_des_one)):
        if i <= 20:
            w_des_one[i] = 0
        else:
            w_des_one[i] = w0

    w_des = np.zeros(len(w_des_one) + len(w_des_two))
    for i in range(len(w_des)):
        if i < len(w_des_one):
            w_des[i] = w_des_one[i]
        else:
            w_des[i] = w_des_two[i - len(w_des_one)]

    tsteps = tsteps + len(w_des_one)

    # Simulation
    # Motor Parameters
    Ke = 0.021199438  # V/rad/s
    Kt = 0.0141937  # Nm/A
    b = 0.0001011492  # Nm/rad/s
    L = 0.00075  # H
    J = 0.00000109445  # kgm^2
    R = 1.56  # ohms
    V_max = 36
    V_min = -V_max

    # simulation time parameters
    tf = dt * (tsteps - 1)
    t_sim = np.linspace(0, tf, tsteps)

    # PID parameters
    error_s = np.zeros(tsteps)
    V_bias = 0
    tau_i = 30
    sum_int = 0.0

    # Initial Conditions
    i0 = 0
    w0 = 0

    # Tunable parameters
    Kp = 0.270727147578817
    Ki = 50.0897752327866
    Kd = 0.000141076220179068
    N = 248711.202620588  # Filter coefficient for filtered derivative

    # Motor input
    V_in = np.zeros(tsteps)

    # ODE Output Storage
    w_s = np.zeros(tsteps)
    i_s = np.zeros(tsteps)

    # DC Motor Model ODEs
    def motor_electrical(i, t, V, w):
        di_dt = (V - R * i - Ke * w) / L
        return di_dt

    def motor_mechanical(w, t, i):
        dw_dt = (Kt * i - b * w) / J
        return dw_dt

    # sim_loop
    for n in range(tsteps - 1):
        # PID Control
        error = w_des[n + 1] - w0
        error_s[n + 1] = error

        sum_int = sum_int + error * dt
        de_dt = (error_s[n + 1] - error_s[n]) / dt
        V_PID = V_bias + Kp * error + Ki * sum_int + (N * Kd) / (1 + N * sum_int) + Kd * de_dt

        # anti-integral windup

        if V_PID > V_max:
            V_PID = V_max
            sum_int = sum_int - error * dt

        if V_PID < V_min:
            V_PID = V_min
            sum_int = sum_int - error * dt

        # PID Data storage
        int_s = sum_int
        V_in[n] = V_PID
        # Motor Actuation
        V = V_in[n]
        t_range = [t_sim[n], t_sim[n + 1]]

        i = odeint(motor_electrical, i0, t_range, args=(V, w0))
        i0 = i[-1][0]
        i_s[n + 1] = i0
        w = odeint(motor_mechanical, w0, t_range, args=(i0,))
        w0 = w[-1]
        w_s[n + 1] = w0

    w_final = w_s[len(w_des_one):]
    return w_final


# This function is run for performing the cubic spline interpolations

def Spline(times, rotations, des_times, k=3):
    spline = interpolate.splrep(times, rotations)
    des_rotations = interpolate.splev(des_times, spline)
    return des_rotations


# This class is set up to run the episodes. 

class Iteration():
    def __init__(self, iteration_ID=1, shedding_freq=8.42, num_actions=25, dur_action_min=2.00, 
                 dur_action_one_add=0.00, CFD_timestep=5e-4, dur_ramps=0.05, free_stream_vel=1.5, 
                 sampling_periods=1.00, CFD_timestep_spacing=5):

        self.iteration_ID = iteration_ID
        self.shedding_freq = shedding_freq
        self.num_actions = num_actions
        self.dur_action_min = dur_action_min
        self.dur_action_one_add = dur_action_one_add
        self.CFD_timestep = CFD_timestep
        self.dur_ramps = dur_ramps
        self.free_stream_vel = free_stream_vel
        self.sampling_periods = sampling_periods
        self.CFD_timestep_spacing = CFD_timestep_spacing

        self.states = []
        self.state = self.reset_state()
        self.actions = []
        self.values = []
        self.mus = []
        self.stds = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.next_states = []

        self.front_cyl_RPS = []
        self.top_cyl_RPS = []
        self.bot_cyl_RPS = []

        self.front_cyl_RPS_PI = []
        self.top_cyl_RPS_PI = []
        self.bot_cyl_RPS_PI = []

        self.top_sens_values = []
        self.mid_sens_values = []
        self.bot_sens_values = []

        self.state_front_cyl_offset = [0,]
        self.state_front_cyl_amp = []
        self.state_front_cyl_phase = []
        self.state_front_cyl_freq = []

        self.state_top_cyl_offset = [0,]
        self.state_top_cyl_amp = []
        self.state_top_cyl_phase = []
        self.state_top_cyl_freq = []

        self.state_bot_cyl_offset = [0,]
        self.state_bot_cyl_amp = []
        self.state_bot_cyl_phase = []
        self.state_bot_cyl_freq = []
        
        self.CFD_timesteps_actions=[]

        self.action_counter = 0
        self.time_step_start = 1

        self.shedding_period = 1 / self.shedding_freq

        self.CFD_timesteps_period = int(np.ceil(self.shedding_period / self.CFD_timestep))
        self.CFD_timesteps_action_min = int(np.ceil(self.CFD_timesteps_period * self.dur_action_min))
        self.CFD_timesteps_action_one_add = int(np.ceil(self.CFD_timesteps_period * self.dur_action_one_add))
        self.CFD_timesteps_ramp = int(np.ceil(self.CFD_timesteps_period * self.dur_ramps))

    # The state is reset here during the start of episodes
    def reset_state(self):
        
        top_sens_var = 0.1635
        mid_sens_var = 0.1700
        bot_sens_var = 0.1481
        top_sens_state = 5.0 * top_sens_var - 1.0
        mid_sens_state = 5.0 * mid_sens_var - 1.0
        bot_sens_state = 5.0 * bot_sens_var - 1.0

        front_mot_state_offset = 0.00
        top_mot_state_offset = 0.00
        bot_mot_state_offset = 0.00

        state = np.array([top_sens_state, mid_sens_state, bot_sens_state,
                          front_mot_state_offset, top_mot_state_offset, bot_mot_state_offset])
        return state

    # The reward is calculated here using the J_fluc and J_act
    def calculate_reward(self):
        
        if len(self.top_sens_values) >= (self.CFD_timesteps_period * self.sampling_periods / self.CFD_timestep_spacing):
            sampling_timesteps = int(self.CFD_timesteps_period * self.sampling_periods / self.CFD_timestep_spacing)
        else:
            sampling_timesteps = int(len(self.top_sens_values))

        top_sens_var = np.var(self.top_sens_values[-sampling_timesteps:])
        mid_sens_var = np.var(self.mid_sens_values[-sampling_timesteps:])
        bot_sens_var = np.var(self.bot_sens_values[-sampling_timesteps:])

        J_fluc = np.mean([top_sens_var, mid_sens_var, bot_sens_var])
        J_fluc = J_fluc / (self.free_stream_vel ** 2)

        J_act = 0

        if len(self.front_cyl_RPS_PI) >= (self.CFD_timesteps_actions[-1] - self.CFD_timesteps_ramp):
            sampling_timesteps = int(self.CFD_timesteps_actions[-1] - self.CFD_timesteps_ramp)
        else:
            sampling_timesteps = int(len(self.front_cyl_RPS_PI))

        for i in range(sampling_timesteps):
            J_act += self.front_cyl_RPS_PI[-(i + 1)] ** 2
            J_act += self.top_cyl_RPS_PI[-(i + 1)] ** 2
            J_act += self.bot_cyl_RPS_PI[-(i + 1)] ** 2

        J_act = np.sqrt(J_act / (3 * sampling_timesteps))
        J_act = J_act / self.free_stream_vel * 0.01

        J_fluc = np.tanh(12.2*J_fluc) 
        J_act = np.tanh(0.7*J_act)
        J_tot = J_fluc + J_act

        J_tot_max = 2.0

        reward = -1 * J_tot / J_tot_max
        reward = np.array([reward])

        return reward

    # The state of the PPO agent in the environment is calculated which is fed to get the action
    def calculate_state(self):
        
        if len(self.top_sens_values) >= (self.CFD_timesteps_period * self.sampling_periods / self.CFD_timestep_spacing):
            sampling_timesteps = int(self.CFD_timesteps_period * self.sampling_periods / self.CFD_timestep_spacing)
        else:
            sampling_timesteps = int(len(self.top_sens_values))

        top_sens_var = np.var(self.top_sens_values[-sampling_timesteps:])
        mid_sens_var = np.var(self.mid_sens_values[-sampling_timesteps:])
        bot_sens_var = np.var(self.bot_sens_values[-sampling_timesteps:])

        top_sens_state = 5.0 * top_sens_var - 1.0
        mid_sens_state = 5.0 * mid_sens_var - 1.0
        bot_sens_state = 5.0 * bot_sens_var - 1.0
        
        if top_sens_state >= 1.5:
            top_sens_state=1.5
        if mid_sens_state >= 1.5:
            mid_sens_state=1.5
        if bot_sens_state >= 1.5:
            bot_sens_state=1.5

        front_mot_state_offset = self.state_front_cyl_offset[-1]
        top_mot_state_offset = self.state_top_cyl_offset[-1]
        bot_mot_state_offset = self.state_bot_cyl_offset[-1]

        state = np.array([top_sens_state, mid_sens_state, bot_sens_state,
                          front_mot_state_offset, top_mot_state_offset, bot_mot_state_offset])
        return state


    # This function uses the action generated by the RL to output the motor data for the CFD. 
    
    def calculate_mot_data(self, action):
        
        print(f'Calculating motor data iteration{self.iteration_ID}-action{action}')

        action_clipped = np.zeros(len(action))

        for i in range(len(action_clipped)):
            if action[i] > 1.75:
                action_clipped[i] = 1.75
            elif action[i] < -1.75:
                action_clipped[i] = -1.75
            else:
                action_clipped[i] = action[i]

        self.state_front_cyl_offset.append(action_clipped[0]/ 1.75)
        self.state_front_cyl_amp.append(action_clipped[1] / 1.75)
        self.state_front_cyl_phase.append(action_clipped[2] / 1.75)
        self.state_front_cyl_freq.append(action_clipped[3] / 1.75)

        self.state_top_cyl_offset.append(action_clipped[4]/ 1.75)
        self.state_top_cyl_amp.append(action_clipped[5] / 1.75)
        self.state_top_cyl_phase.append(action_clipped[6] / 1.75)
        self.state_top_cyl_freq.append(action_clipped[7] / 1.75)

        self.state_bot_cyl_offset.append(action_clipped[8]/ 1.75)
        self.state_bot_cyl_amp.append(action_clipped[9] / 1.75)
        self.state_bot_cyl_phase.append(action_clipped[10] / 1.75)
        self.state_bot_cyl_freq.append(action_clipped[11] / 1.75)
        
        front_cyl_offset = action_clipped[0] * 570
        front_cyl_amp = action_clipped[1] * 570
        front_cyl_phase = action_clipped[2] * 1.79
        
        if action_clipped[3] <=0.329:
            front_cyl_freq=0.0
        else:
            front_cyl_freq=-2.405*action_clipped[3]+9.21

        top_cyl_offset = action_clipped[4] * 570
        top_cyl_amp = action_clipped[5] * 570
        top_cyl_phase = action_clipped[6] * 1.79
        
        if action_clipped[7] <=0.329:
            top_cyl_freq=0.0
        else:
            top_cyl_freq=-2.405*action_clipped[7]+9.21
            
        bot_cyl_offset = action_clipped[8] * 570
        bot_cyl_amp = action_clipped[9] * 570
        bot_cyl_phase = action_clipped[10] * 1.79
        
        if action_clipped[11] <=0.329:
            bot_cyl_freq=0.0
        else:
            bot_cyl_freq=-2.405*action_clipped[11]+9.21
        
        # front_cyl_offset_ramp = action_clipped[0] * 340
        # top_cyl_offset_ramp = action_clipped[4] * 340
        # bot_cyl_offset_ramp = action_clipped[8] * 340
        
        # front_cyl_offset=front_cyl_offset_ramp + (self.state_front_cyl_offset[-1]*1000)
        # top_cyl_offset=top_cyl_offset_ramp + (self.state_top_cyl_offset[-1]*1000)
        # bot_cyl_offset=bot_cyl_offset_ramp + (self.state_bot_cyl_offset[-1]*1000)
        
        # if front_cyl_offset > 1000:
        #     front_cyl_offset=1000
        # if front_cyl_offset <-1000:
        #     front_cyl_offset=-1000
        # if top_cyl_offset > 1000:
        #     top_cyl_offset=1000
        # if top_cyl_offset <-1000:
        #     top_cyl_offset=-1000
        # if bot_cyl_offset > 1000:
        #     bot_cyl_offset=1000
        # if bot_cyl_offset <-1000:
        #     bot_cyl_offset=-1000
            
        # self.state_front_cyl_offset.append(front_cyl_offset/1000)
        # self.state_top_cyl_offset.append(top_cyl_offset/1000)
        # self.state_bot_cyl_offset.append(bot_cyl_offset/1000)
            
        if front_cyl_freq <= 5.0:
            front_cyl_freq = 0.0
            front_cyl_phase=0.0
            front_cyl_amp=0.0
        if top_cyl_freq <= 5.0:
            top_cyl_freq = 0.0
            top_cyl_phase=0.0
            top_cyl_amp=0.0
        if bot_cyl_freq <= 5.0:
            bot_cyl_freq = 0.0
            bot_cyl_phase=0.0
            bot_cyl_amp=0.0
        
        CFD_timesteps_action=self.CFD_timesteps_action_min

        if front_cyl_freq > 5.0:
            front_cyl_timesteps_action = self.CFD_timesteps_period*self.shedding_freq/front_cyl_freq
            
            if front_cyl_timesteps_action > CFD_timesteps_action:
                CFD_timestep_action=front_cyl_timesteps
        
        if top_cyl_freq > 5.0:
            top_cyl_timesteps_action = self.CFD_timesteps_period*self.shedding_freq/top_cyl_freq
            
            if top_cyl_timesteps_action > CFD_timesteps_action:
                CFD_timestep_action=top_cyl_timesteps
        
        if bot_cyl_freq > 5.0:
            bot_cyl_timesteps_action = self.CFD_timesteps_period*self.shedding_freq/bot_cyl_freq
            
            if bot_cyl_timesteps_action > CFD_timesteps_action:
                CFD_timestep_action=bot_cyl_timesteps
                
        CFD_timesteps_action=int(CFD_timesteps_action)
        
        if (CFD_timesteps_action % self.CFD_timestep_spacing) != 0:
            CFD_timesteps_action=CFD_timesteps_action + self.CFD_timestep_spacing - (CFD_timesteps_action % self.CFD_timestep_spacing)
        
        self.CFD_timesteps_actions.append(CFD_timesteps_action)
        
        des_times = np.zeros(self.CFD_timesteps_actions[-1])
        
        for i in range(len(des_times)):
            des_times[i] = self.CFD_timestep * i

        times = np.zeros(10 + self.CFD_timesteps_actions[-1])

        front_cyl_RPS_temp = np.zeros(len(times))
        top_cyl_RPS_temp = np.zeros(len(times))
        bot_cyl_RPS_temp = np.zeros(len(times))

        for i in range(len(times)):
            if i < 5 and len(self.front_cyl_RPS_PI) > 5:
                times[i] = self.CFD_timestep * (-5 + i)
                front_cyl_RPS_temp[i] = self.front_cyl_RPS_PI[-5 + i]
                top_cyl_RPS_temp[i] = self.top_cyl_RPS_PI[-5 + i]
                bot_cyl_RPS_temp[i] = self.bot_cyl_RPS_PI[-5 + i]
            elif i < 5 and len(self.front_cyl_RPS_PI) <= 5:
                times[i] = self.CFD_timestep * (-5 + i)
                front_cyl_RPS_temp[i] = 0
                top_cyl_RPS_temp[i] = 0
                bot_cyl_RPS_temp[i] = 0
            elif i >= 5:
                times[i] = self.CFD_timestep * self.CFD_timesteps_ramp + (i - 5) * self.CFD_timestep
                front_cyl_RPS_temp[i] = front_cyl_offset + front_cyl_amp * np.sin(
                    2 * 3.14 * front_cyl_freq * (i - 5) * self.CFD_timestep + front_cyl_phase)
                top_cyl_RPS_temp[i] = top_cyl_offset + top_cyl_amp * np.sin(
                    2 * 3.14 * top_cyl_freq * (i - 5) * self.CFD_timestep + top_cyl_phase)
                bot_cyl_RPS_temp[i] = bot_cyl_offset + bot_cyl_amp * np.sin(
                    2 * 3.14 * bot_cyl_freq * (i - 5) * self.CFD_timestep + bot_cyl_phase)

                if front_cyl_RPS_temp[i] > 1000:
                    front_cyl_RPS_temp[i] = 1000
                if front_cyl_RPS_temp[i] < -1000:
                    front_cyl_RPS_temp[i] = -1000
                if top_cyl_RPS_temp[i] > 1000:
                    top_cyl_RPS_temp[i] = 1000
                if top_cyl_RPS_temp[i] < -1000:
                    top_cyl_RPS_temp[i] = -1000
                if bot_cyl_RPS_temp[i] > 1000:
                    bot_cyl_RPS_temp[i] = 1000
                if bot_cyl_RPS_temp[i] < -1000:
                    bot_cyl_RPS_temp[i] = -1000

        front_cyl_RPS_temp = Spline(times, front_cyl_RPS_temp, des_times)
        top_cyl_RPS_temp = Spline(times, top_cyl_RPS_temp, des_times)
        bot_cyl_RPS_temp = Spline(times, bot_cyl_RPS_temp, des_times)


        self.front_cyl_RPS.extend(front_cyl_RPS_temp)
        self.top_cyl_RPS.extend(top_cyl_RPS_temp)
        self.bot_cyl_RPS.extend(bot_cyl_RPS_temp)

        if len(self.front_cyl_RPS_PI) == 0:
            front_w0 = 0
            top_w0 = 0
            bot_w0 = 0
        else:
            front_w0 = self.front_cyl_RPS_PI[-1]
            top_w0 = self.top_cyl_RPS_PI[-1]
            bot_w0 = self.bot_cyl_RPS_PI[-1]

        front_cyl_RPS_temp = PI_Motor(front_cyl_RPS_temp, self.CFD_timesteps_actions[-1], self.CFD_timestep, front_w0)
        top_cyl_RPS_temp = PI_Motor(top_cyl_RPS_temp, self.CFD_timesteps_actions[-1], self.CFD_timestep, top_w0)
        bot_cyl_RPS_temp = PI_Motor(bot_cyl_RPS_temp, self.CFD_timesteps_actions[-1], self.CFD_timestep, bot_w0)

        self.front_cyl_RPS_PI.extend(front_cyl_RPS_temp)
        self.top_cyl_RPS_PI.extend(top_cyl_RPS_temp)
        self.bot_cyl_RPS_PI.extend(bot_cyl_RPS_temp)

        mot_data = {'revolutions': [front_cyl_RPS_temp, top_cyl_RPS_temp, bot_cyl_RPS_temp]
            , 'freq': [0, 0, 0], 'amp': [0, 0, 0], 'offset': [0, 0, 0], 'phase': [0, 0, 0]}

        return mot_data
    
    # This function runs the iterations/episodes basically
    
    def run_iteration(self):
        state = self.reset_state()
        for actions in range(num_actions):
            self.action_counter += 1

            action, value, dist, mu, std = ppo_agent._get_action(state)

            log_prob = dist.log_prob(action)
            log_prob = log_prob.detach().numpy()
            action = np.array(action)
            value = value.detach().numpy()
            mu = mu.detach().numpy()
            std = std.detach().numpy()

            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.mus.append(mu)
            self.stds.append(std)

            mot_data = self.calculate_mot_data(action)

            time_step_end = self.time_step_start + self.CFD_timesteps_actions[-1] - 1

            vel_data = CFD_Run(self.iteration_ID, self.action_counter, self.time_step_start, time_step_end, mot_data)

            vel_data_top = vel_data['top']
            vel_data_mid = vel_data['mid']
            vel_data_bot = vel_data['bot']

            self.top_sens_values.extend(vel_data_top)
            self.mid_sens_values.extend(vel_data_mid)
            self.bot_sens_values.extend(vel_data_bot)

            reward = self.calculate_reward()
            state = self.calculate_state()

            if self.action_counter < self.num_actions:
                is_terminal = np.array([1.0])
            else:
                is_terminal = np.array([0.0])

            self.rewards.append(reward)
            self.next_states.append(state)
            self.is_terminals.append(is_terminal)

            self.time_step_start = time_step_end + 1

    def run_action(self):
        self.action_counter += 1
        print(f'{self.iteration_ID}-{self.action_counter} -- '
              f'Action: {self.action} Value: {self.value}'
              f'Dist: {self.dist} Mu: {self.mu} STD: {self.std}')

        log_prob = self.dist.log_prob(self.action)
        log_prob = log_prob.detach().numpy()
        action = np.array(self.action)
        value = self.value.detach().numpy()
        mu = self.mu.detach().numpy()
        std = self.std.detach().numpy()

        self.states.append(self.state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.mus.append(mu)
        self.stds.append(std)

        mot_data = self.calculate_mot_data(action)

        time_step_end = self.time_step_start + self.CFD_timesteps_actions[-1] - 1

        vel_data = CFD.cfd.RL_run_CFD(iteration=self.iteration_ID, action=self.action_counter,
                                      t_start=self.time_step_start, t_end=time_step_end,
                                      motor_params=mot_data)
        # Manual hotfix for issues on cluster
        while vel_data is None:
            vel_data = CFD.cfd.RL_get_sensor_data(iteration=self.iteration_ID,
                                                  action=self.action_counter,
                                                  t_start=self.time_step_start,
                                                  t_end=time_step_end)

        while len(vel_data['top']) < 3:
            vel_data = CFD.cfd.RL_get_sensor_data(iteration=self.iteration_ID,
                                                  action=self.action_counter,
                                                  t_start=self.time_step_start,
                                                  t_end=time_step_end)
        print(vel_data)

        vel_data_top = vel_data['top']
        vel_data_mid = vel_data['mid']
        vel_data_bot = vel_data['bot']

        self.top_sens_values.extend(vel_data_top)
        self.mid_sens_values.extend(vel_data_mid)
        self.bot_sens_values.extend(vel_data_bot)

        reward = self.calculate_reward()
        self.state = self.calculate_state()
        if self.action_counter < self.num_actions:
            is_terminal = np.array([1.0])
        else:
            is_terminal = np.array([0.0])

        self.rewards.append(reward)
        self.next_states.append(self.state)
        self.is_terminals.append(is_terminal)

        self.time_step_start = time_step_end + 1

    def update_action(self, action, value, dist, mu, std):
        self.action = action
        self.value = value
        self.dist = dist
        self.mu = mu
        self.std = std

    # This function saves the iteration data into pickle files
    def save_iteration(self):
        iteration_results = {'iteration_ID': self.iteration_ID, 'states': self.states, 'actions': self.actions,
                             'rewards': self.rewards, 'mus': self.mus, 'stds': self.stds, 'values': self.values,
                             'is_terminals': self.is_terminals, 'log_probs': self.log_probs,
                             'front_cyl_RPS': self.front_cyl_RPS, 'top_cyl_RPS': self.top_cyl_RPS,
                             'bot_cyl_RPS': self.bot_cyl_RPS, 'front_cyl_RPS_PI': self.front_cyl_RPS_PI,
                             'top_cyl_RPS_PI': self.top_cyl_RPS_PI, 'bot_cyl_RPS_PI': self.bot_cyl_RPS_PI,
                             'top_sens_values': self.top_sens_values, 'mid_sens_values': self.mid_sens_values,
                             'bot_sens_values': self.bot_sens_values, 'CFD_timesteps_actions':self.CFD_timesteps_actions}

        filename = 'data_iteration_' + str(self.iteration_ID) + '.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(iteration_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _run_action(iteration):
    # Read in the Pickle file to load in action data from PPO agent.
    data = {}
    for n, name in enumerate(['action', 'value', 'dist', 'mu', 'std']):
        fname = f'{iteration.iteration_ID}-{name}.pickle'
        with open(fname, 'rb') as f:
            data[name] = pickle.load(f)
    action = data['action']
    value = data['value']
    dist = data['dist']
    mu = data['mu']
    std = data['std']

    # Update the Iteration class' action parameters with data we just loaded.
    iteration.update_action(action=action, value=value, dist=dist, mu=mu, std=std)

    iteration.run_action()

    return iteration


##################     MAIN LOOP BEGINS HERE     ###############################

# We define the parameters for both actor and critic NNs

obs_dim = 6
act_dim = 12
gamma = 0.99
lamda = 0.10
entropy_coef = 0.001
epsilon = 0.25
num_updates = 1
num_epochs = 10
batch_size = 250
actor_lr = 1e-4
critic_lr = 1e-4
lr_manual_bool = True
actor_lr_max = 1e-4
actor_lr_min = 1e-4
critic_lr_max = 3.5e-4
critic_lr_min = 1e-4
total_critic_losses = []
total_actor_losses = []

# We feed the NN parameters to the PPO agent class
ppo_agent = PPO_Agent(obs_dim=obs_dim, act_dim=act_dim, gamma=gamma, lamda=lamda, entropy_coef=entropy_coef,
                      epsilon=epsilon, num_epochs=num_epochs, batch_size=batch_size,
                      actor_lr=actor_lr, critic_lr=critic_lr)

# We define the required CFD and RL defining parameters for the PPO agent here

shedding_freq = 8.42
dur_action_min = 2.00
dur_action_one_add = 0.00
CFD_timestep = 5e-4
CFD_timestep_spacing = 5
dur_ramps = 0.05
num_actions = 25
num_policies = 40
num_iterations = 10
free_stream_vel = 1.5
sampling_periods = 1.00
load_weights = False
policy_num_load_weights = 0


# This is used if we load the weights. Usually, this is not used though unless the simulation gets interrupted
# or something
def main():
    if load_weights:
        filename_actor = 'Actor_Policy_Number_' + str(policy_num_load_weights)
        filename_critic = 'Critic_Policy_Number_' + str(policy_num_load_weights)
        ppo_agent._load_weights(filename_actor, filename_critic)

    # Main loop that goes through the policies and episodes/iterations. We also save the total rewards for every episode
    # here into a pickle file

    for policy in range(policy_num_load_weights, num_policies):
        actor_lr = (actor_lr_max - actor_lr_min) / (num_policies) * (num_policies - policy) + actor_lr_min
        critic_lr = (critic_lr_max - critic_lr_min) / (num_policies) * (num_policies - policy) + critic_lr_min
        Iterations = []
        for iteration in range(num_iterations):
            iteration_ID = num_iterations * policy + iteration + 1
            Iterations.append(Iteration(iteration_ID=iteration_ID, shedding_freq=shedding_freq, num_actions=num_actions,
                                        dur_action_min=dur_action_min, dur_action_one_add=dur_action_one_add,
                                        CFD_timestep=CFD_timestep, dur_ramps=dur_ramps, free_stream_vel=free_stream_vel,
                                        sampling_periods=sampling_periods, CFD_timestep_spacing=CFD_timestep_spacing))

        if os.path.exists(f'policy-{policy + 1}'):
            os.chdir(f'policy-{policy + 1}')
        else:
            os.mkdir(f'policy-{policy + 1}')
            os.chdir(f'policy-{policy + 1}')

        # Compute actions for all iterations simultaneously
        for action in range(num_actions):
            for iteration in Iterations:
                print('-' * 50)
                print(f'State: {iteration.iteration_ID}-{action + 1}-{iteration.state}')
                print('-' * 50)
            # KEY ISSUE #
            # Must dump action data retrieved from PPO agent into Pickle file.
            # This pickle file is then read in by the Iteration, to run its action.
            actions = {iteration.iteration_ID: ppo_agent._get_action(iteration.state) for iteration in Iterations}
            for iteration in Iterations:
                for n, fname in enumerate(['action', 'value', 'dist', 'mu', 'std']):
                    with open(f'{iteration.iteration_ID}-{fname}.pickle', 'wb') as f:
                        pickle.dump(actions[iteration.iteration_ID][n], f, protocol=pickle.HIGHEST_PROTOCOL)

            # Run action for all Iterations. (The same action number).
            with mp.Pool() as p:
                Iterations = p.map(_run_action, Iterations)

        for iteration in range(num_iterations):
            Iterations[iteration].save_iteration()
        for iteration in range(num_iterations):
            ppo_agent.states.extend(Iterations[iteration].states)
            ppo_agent.actions.extend(Iterations[iteration].actions)
            ppo_agent.rewards.extend(Iterations[iteration].rewards)
            ppo_agent.is_terminals.extend(Iterations[iteration].is_terminals)
            ppo_agent.log_probs.extend(Iterations[iteration].log_probs)
            ppo_agent.next_states.extend(Iterations[iteration].next_states)

        for update in range(num_updates):
            ppo_agent.values = []
            for i in range(len(ppo_agent.states)):
                value = ppo_agent.critic.forward(torch.FloatTensor(ppo_agent.states[i]))
                value = value.detach().numpy()
                ppo_agent.values.append(value)
            next_state = ppo_agent.next_states[-1]
            value = ppo_agent.critic.forward(torch.FloatTensor(next_state))
            value = value.detach().numpy()
            ppo_agent.values.append(value)
            ppo_agent._update_weights(lr_manual_bool, actor_lr, critic_lr)

        ppo_agent._save_weights((policy + 1))
        ppo_agent.clear_memory()
        print('Weights Updated')

        critic_losses = ppo_agent.critic_losses[-num_updates:]
        actor_losses = ppo_agent.actor_losses[-num_updates:]
        total_critic_losses.append(critic_losses)
        total_actor_losses.append(actor_losses)
        losses_dictionary = {'actor_losses': total_actor_losses, 'critic_losses': total_critic_losses}
        losses_filename = 'actor_critic_losses_' + str(policy + 1) + '.pickle'
        with open(losses_filename, 'wb') as handle:
            pickle.dump(losses_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

        os.chdir('../')

if __name__ == '__main__':
    main()

