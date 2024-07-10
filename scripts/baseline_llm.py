import os
import re
import jinja2
import numpy as np
from openai import OpenAI
from air_hockey_llm.environments.air_hockey_hit import IiwaPositionHit
from air_hockey_llm.baseline_agent_instruct import BaselineAgent


def query_model(port, chat_history):
    api_key = "EMPTY"
    api_base = f"http://localhost:{port}/v1"
    client = OpenAI(api_key=api_key, base_url=api_base)
    model = "deepseek-ai/deepseek-coder-6.7b-instruct"

    completion = client.chat.completions.create(
        model=model,
        messages=chat_history,
        max_tokens=2048,
        temperature=0.1
    )

    chat_history.append({"role": "assistant",
                         "content": completion.choices[0].message.content})

    return completion.choices[0].message.content


def process_answer(answer):
    code_block_regex = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    code_blocks = code_block_regex.findall(answer)

    if code_blocks:
        full_code = "\n".join(code_blocks)
    else:
        full_code = None

    exec(full_code)

    if "def " not in code_blocks[0]:
        return code_blocks[0]
    else:
        function_name = code_blocks[0].split("def ")[1].split("(")[0]
        return eval(function_name)


def main(port, prompt_dir):
    viewer_params = {'camera_params': {'static': dict(distance=3.0, elevation=-45.0, azimuth=90.0,
                                                      lookat=(0., 0., 0.))},
                     'width': 1440, 'height': 810,
                     'default_camera_mode': 'static',
                     'hide_menu_on_startup': True}

    env = IiwaPositionHit(interpolation_order=-1, viewer_params=viewer_params, horizon=200)
    # env.puck_init_pos = np.array([-0.5, 0.3])

    agent = BaselineAgent(env_info=env.env_info, agent_id=1, only_tactic='hit', max_hit_velocity=1.0)

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(prompt_dir),
        autoescape=jinja2.select_autoescape(enabled_extensions=('jinja'), default_for_string=True)
    )

    n_episodes = 10
    env.reset()

    system_prompt = jinja_env.get_template("system.jinja").render()
    initial_prompt = jinja_env.get_template("initial.jinja").render()
    chat_history = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_prompt}]
    answer = query_model(port=port, chat_history=chat_history)
    print("Answer: ", answer)
    compute_angle_and_velocity = process_answer(answer)
    prev_hit_angle = list()
    prev_puck_final_pos = list()
    prev_dist_to_goal = list()
    angle_offset = 0.

    for i in range(n_episodes):
        done = False
        # env.puck_init_pos = np.random.rand(2) * (env.hit_range[:, 1] - env.hit_range[:, 0]) + env.hit_range[:, 0]
        # env.puck_init_vel = np.random.rand(2) * 0.2

        env.puck_init_pos = np.array([-0.5, -0.3])
        env.puck_init_vel = np.array([0., 0.1])
        obs = env.reset()
        puck_pos = obs[:2] - np.array([1.51, 0.])
        puck_vel = obs[3:5]
        goal_pos = np.array([0.938, 0.0])

        angle, velocity_scale = compute_angle_and_velocity(puck_init_pos=puck_pos, puck_init_vel=puck_vel, goal_pos=goal_pos)
        # angle_offset = compute_angle_offset(angle, np.array(prev_hit_angle), np.array(prev_puck_final_pos), goal_pos)
        angle += angle_offset
        agent.set_instruction({'hit_angle': angle, 'hit_velocity': velocity_scale})
        
        print(f"Hit Angle: {angle}, Hit Velocity Scale: {velocity_scale}, Offset: {angle_offset}")

        n_steps = 0
        hit_puck_vel = 0.
        while n_steps < env.info.horizon:
            action = agent.draw_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            n_steps += 1
            if np.linalg.norm(hit_puck_vel) == 0. and obs[0] >= 1.51:
                hit_puck_vel = obs[3:5]
            if done:
                break

        puck_final_pos = obs[:2] - np.array([1.51, 0.])
        dist_to_goal = np.linalg.norm(puck_final_pos - goal_pos)

        prev_hit_angle.append(angle)
        prev_puck_final_pos.append(puck_final_pos)
        prev_dist_to_goal.append(dist_to_goal)

        if dist_to_goal < 0.1:
            print("Goal! Start a new trial.")
        else:
            continue_prompt = jinja_env.get_template("continue.jinja").render(
                hit_angle=prev_hit_angle, puck_final_position=prev_puck_final_pos, dist_to_goal=prev_dist_to_goal,
                goal_pos=goal_pos)

            print("#########################################")
            print("New prompt: ", continue_prompt)
            chat_history.append({"role": "user", "content": continue_prompt})
            answer = query_model(port=port, chat_history=chat_history)
            code_snippet = process_answer(answer)
            print("Answer: ", answer)
            class CodeSnippet:
                angle_offset = 0.
                exec(code_snippet)
            angle_offset = CodeSnippet.angle_offset
            chat_history.pop(-1)
            chat_history.pop(-1)
            print("Angle Offset: ", angle_offset)


if __name__ == '__main__':
    port = 8080
    prompt_dir = "prompts/air_hockey"
    # answer = query_model(port=port, file_dir=prompt_file)
    # with open(f"answer_{port}.txt", "w") as f:
    #     f.write(answer)

    main(port, prompt_dir)
