import torch

def command2NumericAction(command):
    #self.available_controls = params.get('available_controls',
    # ['turnLeft', 'turnRight', 'forwards', 'backwards', 'strafeLeft', 'strafeRight', 'lookDown', 'lookUp']
    action =  torch.zeros(8)
    if command == 'turnLeft':
        action[0] = 1
    elif command == 'turnRight':
        action[1] = 1
    elif command == 'forwards':
        action[2] = 1
    elif command == 'backwards':
        action[3] = 1
    elif command == 'strafeLeft':
        action[4] = 1
    elif command == 'strafeRight':
        action[5] = 1
    elif command == 'lookDown':
        action[6] = 1
    elif command == 'lookUp':
        action[7] = 1
    elif command == 'Idle':
        action = torch.zeros(8) 
    else:
        print ("Warning, unrecognized action")
        pass
    return action.numpy()

def action2Command(action):
    command = 'Idle'
    if action[0] > 0:
        command = 'turnLeft'
    elif action[1]  > 0:
            command = 'turnRight'
    elif action[2] > 0:
        command = 'forwards'
    elif action[3] > 0:
        command = 'backwards'
    elif action[4] > 0:
        command = 'strafeLeft'
    elif action[5] > 0:
        command = 'strafeRight'
    elif action[6] > 0:
        command = 'lookDown'
    elif action[7] > 0:
        command = 'lookUp'
    else:
        command = 'Idle'
    return command

def action2Command9(action):
    command = 'Idle'
    idx = torch.argmax(action).item()
    if idx == 0:
        command = 'turnLeft'
    elif idx == 1:
            command = 'turnRight'
    elif idx == 2:
        command = 'forwards'
    elif idx == 3:
        command = 'backwards'
    elif idx == 4:
        command = 'strafeLeft'
    elif idx == 5:
        command = 'strafeRight'
    elif idx == 6:
        command = 'lookDown'
    elif idx == 7:
        command = 'lookUp'
    elif idx == 8:
        command = 'Idle'
    else:
        command = 'Idle'
    return command
