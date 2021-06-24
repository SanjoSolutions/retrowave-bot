from tensorflow.keras.initializers import Zeros
from tensorflow.keras.layers import Dense, concatenate, Conv2D, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from mss import mss
import numpy as np
import cv2 as cv
import pyautogui
from ctypes import *
import time
import random
from pymem import Pymem
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
import tensorflow as tf

MONITOR_INDEX = 1
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
NUMBER_OF_COLOR_CHANNELS = 1

actions = (
    # ('w', 'shiftleft',),
    ('w',),
    # ('a',),
    # ('s',),
    # ('d',),
    # ('space',),
    ('w', 'a'),
    ('w', 'd'),
    # ('s', 'a'),
    # ('s', 'd'),
    # ('w', 'a', 'shiftleft'),
    # ('w', 'd', 'shiftleft'),
    # ('w', 'a', 'space'),
    # ('w', 'd', 'space'),
    # ('a', 'space'),
    # ('d', 'space'),
    # tuple()
)

hwnd = cdll.user32.FindWindowW(None, 'Retrowave')


pressed_keys = set()


def release_pressed_keys():
    global pressed_keys
    for key in pressed_keys:
        pyautogui.keyUp(key)
    pressed_keys = set()


def read_pointer(process: Pymem, base_address, offsets):
    address = process.read_ulong(base_address)
    for offset in offsets:
        address += offset
        address = process.read_ulong(address)
    return address


def read_score_address(process):
    modules = list(process.list_modules())
    module = next(module for module in modules if module.name == 'UnityPlayer.dll')
    module_base_address = module.lpBaseOfDll
    score_address = read_pointer(
        process,
        module_base_address + 0x01176F48,
        (
            0x28,
            0x38,
            0x4,
            0x10,
            0x10,
            0xC
        )
    ) + 0x24
    return score_address


def determine_is_game_overlay_rendered_address(process):
    modules = list(process.list_modules())
    module = next(module for module in modules if module.name == 'gameoverlayrenderer.dll')
    module_base_address = module.lpBaseOfDll
    is_game_overlay_rendered_address = module_base_address + 0x140948
    return is_game_overlay_rendered_address


def read_score(process):
    score_address = read_score_address(process)
    score = process.read_float(score_address)
    return score


def read_is_game_overlay_rendered(process, is_game_overlay_rendered_address):
    is_game_overlay_rendered = process.read_int(is_game_overlay_rendered_address)
    return is_game_overlay_rendered


def is_game_overlay_rendered(process, is_game_overlay_rendered_address):
    return read_is_game_overlay_rendered(process, is_game_overlay_rendered_address) != 0


def is_done(process, is_game_overlay_rendered_address):
    return is_game_overlay_rendered(process, is_game_overlay_rendered_address)


process = Pymem('Retrowave.exe')
is_game_overlay_rendered_address = determine_is_game_overlay_rendered_address(process)


# state, action -> value
# state -> action


def create_convolutional_layers():
    image_input_layer = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, NUMBER_OF_COLOR_CHANNELS), name='image')
    convolution_layer_1 = Conv2D(
        filters=32,
        kernel_size=(8, 8),
        strides=4,
        activation='relu',
    )(image_input_layer)
    convolution_layer_2 = Conv2D(
        filters=64,
        kernel_size=(4, 4),
        strides=2,
        activation='relu',
    )(convolution_layer_1)
    convolution_layer_3 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        activation='relu',
    )(convolution_layer_2)
    flatten_convolution_layer_3 = Flatten()(convolution_layer_3)
    return image_input_layer, flatten_convolution_layer_3


model_path = r'D:\\retrowave-bot'
state_and_action_model_path = model_path + r'\state_and_action_to_value'
state_to_action_model_path = model_path + r'\state_to_action'


def create_state_and_action_to_value_model():
    # image_input_layer = KerasLayer(
    #     "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    #     input_shape=(224, 224, 3),
    #     output_shape=(1280,),
    #     dtype=tf.float32,
    #     trainable=False
    # )
    # image_input_layer.shape = TensorShape((None, 1280))
    image_input_layer, flatten_convolution_layer_3 = create_convolutional_layers()
    action_input_layer = Input(shape=(len(actions),), name='action')
    hidden_layer_1 = concatenate([flatten_convolution_layer_3, action_input_layer])
    hidden_layer_2 = Dense(512, activation='relu')(hidden_layer_1)
    output_layer = Dense(1, name='value', kernel_initializer=Zeros())(hidden_layer_2)
    state_and_action_to_value = Model(inputs=[image_input_layer, action_input_layer], outputs=output_layer)
    state_and_action_to_value.compile(
        optimizer=Adam(),
        loss=MeanSquaredError()
    )
    return state_and_action_to_value


state_and_action_to_value = create_state_and_action_to_value_model()
# state_and_action_to_value = load_model(state_and_action_model_path)
# state_and_action_to_value.compile(
#     optimizer=Adam(learning_rate=0.01),
#     loss=MeanSquaredError()
# )


def determine_maximum_value_action(state):
    x_1 = np.array([state] * len(actions))
    x_2 = np.array([action_to_action_embedding_with_caching(action) for action in range(len(actions))])
    predicted_values = state_and_action_to_value.predict({'image': x_1, 'action': x_2})
    index = np.argmax(predicted_values)
    action = index
    return action


def create_state_to_action_model():
    state_to_action_image_input_layer, state_to_action_flatten_convolution_layer_3 = create_convolutional_layers()
    state_to_action_hidden_layer = Dense(512, activation='relu')(state_to_action_flatten_convolution_layer_3)
    state_to_action_output_layer = Dense(len(actions), activation='softmax', kernel_initializer=Zeros())(state_to_action_hidden_layer)
    state_to_action = Model(inputs=state_to_action_image_input_layer, outputs=state_to_action_output_layer)
    state_to_action.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy()
    )
    return state_to_action


state_to_action = create_state_to_action_model()
# state_to_action = load_model(state_to_action_model_path)


def save_models():
    print('Saving models...')
    state_and_action_to_value.save(state_and_action_model_path)
    state_to_action.save(state_to_action_model_path)


def cache(function):
    cache = dict()

    def function_with_cache(*arguments):
        arguments = tuple(arguments)
        if arguments in cache:
            result = cache[arguments]
        else:
            result = function(*arguments)
            cache[arguments] = result
        return result

    return function_with_cache


def action_to_action_embedding(action):
    embedding = [0] * len(actions)
    embedding[action] = 1
    return embedding


action_to_action_embedding_with_caching = cache(action_to_action_embedding)


def play_again():
    pyautogui.click(x=1042, y=1215)
    time.sleep(0.1)
    pyautogui.press('c')


episode_number = 0
exploration_rate = 0


try:
    with mss() as screenshotter:
        monitor = screenshotter.monitors[MONITOR_INDEX]
        if not (monitor['width'] == 2560 and monitor['height'] == 1440):
            raise ValueError(
                "This code has been written for the monitor resolution 2560x1440. " +
                "Please adjust the code if you'd like to use it."
            )
        rect = {
            'left': 1040,
            'top': 511,
            'width': 479,
            'height': 479
        }

        def make_screenshot():
            screenshot = screenshotter.grab(rect)  # RGB
            frame = np.array(screenshot.pixels, dtype=np.uint8)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = np.array(frame, dtype=np.float32)
            frame /= 255.0
            frame = frame.reshape((FRAME_WIDTH, FRAME_HEIGHT, NUMBER_OF_COLOR_CHANNELS))
            return frame

        MAXIMUM_REPLAY_BUFFER_SIZE = 289 * 10e3
        replay_buffer = []


        def add_to_replay_buffer(step):
            global replay_buffer
            number_of_steps_to_add = 1
            if len(replay_buffer) + number_of_steps_to_add >= MAXIMUM_REPLAY_BUFFER_SIZE:
                shrink_replay_buffer_to_size = MAXIMUM_REPLAY_BUFFER_SIZE - number_of_steps_to_add
                replay_buffer = replay_buffer[-shrink_replay_buffer_to_size:]
            replay_buffer.append(step)


        while True:
            i = 1

            while (
                is_done(process, is_game_overlay_rendered_address) or
                read_score(process) > 0
            ):
                time.sleep(1)
            episode_number += 1
            exploration_rate = max(
                0.1,
                (1.0 - 0.25) - float(tf.math.sigmoid(float(episode_number) * (4.0 / 100.0)).numpy())
            )
            print('Episode ' + str(episode_number))
            release_pressed_keys()
            previous_score = 0
            while not is_done(process, is_game_overlay_rendered_address):
                if windll.user32.GetForegroundWindow() != hwnd:
                    release_pressed_keys()
                while windll.user32.GetForegroundWindow() != hwnd:
                    time.sleep(1)
                screenshot = make_screenshot()
                state = screenshot
                # action = random.choice(actions)
                # action = actions[4]  # 77
                if random.random() <= exploration_rate:
                    action = random.choice(list(range(len(actions))))
                else:
                    x = state.reshape((1, FRAME_WIDTH, FRAME_HEIGHT, NUMBER_OF_COLOR_CHANNELS))
                    probabilities = state_to_action(x).numpy()[0]
                    action = int(np.argmax(probabilities))
                print(i)
                i += 1
                action_keys = actions[action]
                score = read_score(process)
                reward = score - previous_score
                step = (state, action, reward)
                add_to_replay_buffer(step)
                keys_to_press = set(action_keys)
                released_keys = set()
                for key in pressed_keys:
                    if key not in keys_to_press:
                        pyautogui.keyUp(key)
                        released_keys.add(key)
                pressed_keys -= released_keys
                for key in action_keys:
                    if key not in pressed_keys:
                        pyautogui.keyDown(key)
                        pressed_keys.add(key)
                previous_score = score

            release_pressed_keys()


            BATCH_SIZE = 32
            NUMBER_OF_TRAINING_SAMPLES = 32 * BATCH_SIZE

            samples = random.sample(replay_buffer, min(NUMBER_OF_TRAINING_SAMPLES, len(replay_buffer)))

            # train models with data of last episode
            print('Training state and action to value model...')

            x_1 = np.array([state for state, action, reward in samples])
            x_2 = np.array([action_to_action_embedding_with_caching(action) for state, action, reward in samples])
            y = np.array([[reward] for state, action, reward in samples])
            state_and_action_to_value.fit(
                x={'image': x_1, 'action': x_2},
                y={'value': y},
                epochs=1000,
                batch_size=BATCH_SIZE,
                callbacks=[
                    EarlyStopping(
                        monitor='loss',
                        mode='min',
                        patience=2,
                        restore_best_weights=True
                    )
                ]
            )

            print('Training state to action model...')

            x = np.array([state for state, action, reward in samples])
            y = np.array([action_to_action_embedding_with_caching(determine_maximum_value_action(state)) for state, action, reward in samples])

            state_to_action.fit(
                x=x,
                y=y,
                epochs=100,
                batch_size=BATCH_SIZE,
                callbacks=[
                    EarlyStopping(
                        monitor='loss',
                        mode='min',
                        patience=2,
                        restore_best_weights=True
                    )
                ]
            )

            save_models()

            play_again()

except KeyboardInterrupt:
    release_pressed_keys()

    save_models()
