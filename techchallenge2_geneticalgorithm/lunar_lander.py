import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
individual=[1,0,3,3,1,1,1,2,1,2,1,2,0,2,1,1,1,2,2,0,0,3,1,1,1,2,3,2,2,3,0,1,2,1,2,0,1,1,0,1,2,0,0,3,2,1,2,3,0,3,3,1,1,3,0,1,3,3,2,1,0,2,1,3,2,3,3,0,3,0,3,1,3,2,2,1,3,1,3,3,1,2,3,0,0,3,0,0,3,3,0,2,2,2,3,0,3,2,1,1,1,3,3,1,0,1,3,0,3,1,3,3,0,0,2,1,3,3,2,0,1,2,1,2,1,3,3,0,2,0,0,3,1,1,2,2,3,1,3,3,0,1,1,3,0,0,3,1,3,2,0,1,2,2,1,2,2,2,0,0,0,0,1,0,3,0,3,0,2,3,0,3,3,0,0,1,3,1,3,0,3,3,0,0,1,1,0,3,1,1,1,2,0,0,1,0,0,2,0,1]
for i in range(len(individual)):
   action = individual[i]  # this is where you would insert your policy
   print ("i", i, "action", action )
   # print( env.action_space)
   # print ("action", action)
   # individual = [env.action_space.sample() for _ in range(200)]
   # individual=[1,0,3,3,1,1,1,2,1,2,1,2,0,2,1,1,1,2,2,0,0,3,1,1,1,2,3,2,2,3,0,1,2,1,2,0,1,1,0,1,2,0,0,3,2,1,2,3,0,3,3,1,1,3,0,1,3,3,2,1,0,2,1,3,2,3,3,0,3,0,3,1,3,2,2,1,3,1,3,3,1,2,3,0,0,3,0,0,3,3,0,2,2,2,3,0,3,2,1,1,1,3,3,1,0,1,3,0,3,1,3,3,0,0,2,1,3,3,2,0,1,2,1,2,1,3,3,0,2,0,0,3,1,1,2,2,3,1,3,3,0,1,1,3,0,0,3,1,3,2,0,1,2,2,1,2,2,2,0,0,0,0,1,0,3,0,3,0,2,3,0,3,3,0,0,1,3,1,3,0,3,3,0,0,1,1,0,3,1,1,1,2,0,0,1,0,0,2,0,1]
   # print ("individual", individual[0])
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      print("terminated", terminated)
      print("truncated", truncated)
      break;
      # observation, info = env.reset()

env.close()