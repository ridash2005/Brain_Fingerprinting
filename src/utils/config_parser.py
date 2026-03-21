import os

def parse_basic_params(): 
    basic_parameters = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config', 'basic_parameters.txt')
    with open(config_path) as f:
        current_key = None
        current_value = []
        inside_list = False
        
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines or comments
                continue
            
            if '=' in line:  # Handle lines with '=' (key-value pairs)
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
    
                # Handle lists (start of a list)
                if value.startswith("["):
                    current_key = key
                    current_value = [value]
                    inside_list = True
                else:
                    # Convert to appropriate type (int, float, or string)
                    if value.isdigit():  # Integer
                        basic_parameters[key] = int(value)
                    elif value.replace('.', '', 1).isdigit():  # Float
                        basic_parameters[key] = float(value)
                    elif value.startswith('"') and value.endswith('"'):  # String
                        basic_parameters[key] = value.strip('"')
                    else:
                        basic_parameters[key] = value  # Fallback for strings without quotes
            elif inside_list:
                # Accumulate list items across lines
                current_value.append(line)
                if line.endswith("]"):  # End of the list
                    inside_list = False
                    list_str = ''.join(current_value).replace('[', '').replace(']', '')
                    list_items = [item.strip().strip('"') for item in list_str.split(',')]
                    basic_parameters[current_key] = list_items
        return basic_parameters
