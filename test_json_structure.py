import json
from types import SimpleNamespace

def load_config(json_file='creds.json'):
    """Load configuration from JSON file and make it accessible like a module"""
    try:
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        
        # Convert dict to object with dot notation access
        def dict_to_obj(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_obj(item) for item in d]
            else:
                return d
        
        return dict_to_obj(config_dict)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

if __name__ == "__main__":
    print("Testing JSON structure...")
    
    try:
        # First, let's see the raw JSON
        with open('creds.json', 'r') as f:
            raw_json = json.load(f)
        
        print("\n📊 Raw JSON structure:")
        print(f"Keys: {list(raw_json.keys())}")
        
        if 'INDICATORS' in raw_json:
            print(f"\n🔍 INDICATORS section:")
            indicators = raw_json['INDICATORS']
            print(f"Type: {type(indicators)}")
            print(f"Keys: {list(indicators.keys())}")
            
            for name, config in indicators.items():
                print(f"\n  {name}:")
                print(f"    Type: {type(config)}")
                print(f"    Keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
                if isinstance(config, dict) and 'timeframes' in config:
                    print(f"    Timeframes: {config['timeframes']}")
                if isinstance(config, dict) and 'params' in config:
                    print(f"    Params: {config['params']}")
        else:
            print("❌ INDICATORS section not found in creds.json")
        
        # Now test the SimpleNamespace conversion
        print("\n🔄 Testing SimpleNamespace conversion...")
        creds = load_config('creds.json')
        
        if creds:
            print("✅ Successfully converted to SimpleNamespace!")
            print(f"creds.INDICATORS type: {type(creds.INDICATORS)}")
            
            if hasattr(creds, 'INDICATORS'):
                print(f"INDICATORS keys: {list(creds.INDICATORS.keys())}")
                
                for name, config in creds.INDICATORS.items():
                    print(f"\n  {name}:")
                    print(f"    Type: {type(config)}")
                    if hasattr(config, 'timeframes'):
                        print(f"    Timeframes: {config.timeframes}")
                    if hasattr(config, 'params'):
                        print(f"    Params: {config.params}")
            else:
                print("❌ creds.INDICATORS not found")
        else:
            print("❌ Failed to convert to SimpleNamespace")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
