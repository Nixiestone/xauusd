"""
Quick Fix & Test Script
=======================
This will help you get everything working quickly

Run: python quick_fix.py
"""

import sys
import os

def main():
    print("="*70)
    print("XAUUSD STRATEGY - QUICK FIX & TEST")
    print("="*70)
    print()
    
    print("Select what you need help with:")
    print("1. Find my broker's gold symbol name")
    print("2. Test Telegram bot connection")
    print("3. Run strategy with sample data (should generate signals)")
    print("4. Configure bot for live trading")
    print()
    
    choice = input("Enter choice (1/2/3/4): ").strip()
    print()
    
    if choice == "1":
        # Check MT5 symbols
        print("Checking MT5 symbols...")
        print("Please wait while I scan your broker's symbols...")
        print()
        
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                print("❌ Cannot connect to MT5")
                print("   Make sure MT5 is running and you're logged in")
                return
            
            print("✓ Connected to MT5")
            account = mt5.account_info()
            if account:
                print(f"  Broker: {account.company}")
                print(f"  Server: {account.server}")
            print()
            
            # Find gold symbols
            symbols = mt5.symbols_get()
            gold_found = []
            
            for s in symbols:
                if any(x in s.name.upper() for x in ['XAU', 'GOLD', 'GLD']):
                    # Test if data available
                    rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_M15, 0, 5)
                    if rates is not None and len(rates) > 0:
                        gold_found.append(s.name)
            
            if gold_found:
                print(f"✅ Found working gold symbols:")
                for sym in gold_found:
                    print(f"   • {sym}")
                print()
                print(f"📝 Use this in your code: '{gold_found[0]}'")
                print()
                print("Update line in xauusd_strategy.py:")
                print(f'   Change "XAUUSD" to "{gold_found[0]}"')
            else:
                print("❌ No working gold symbols found")
                print()
                print("Options:")
                print("1. In MT5, right-click Market Watch → Show All")
                print("2. Find gold in the list and enable it")
                print("3. Or use sample data mode for testing")
            
            mt5.shutdown()
            
        except ImportError:
            print("❌ MetaTrader5 not installed")
            print("   Install: pip install MetaTrader5")
    
    elif choice == "2":
        # Test Telegram
        print("Testing Telegram bot...")
        print()
        
        bot_token = input("Enter your bot token: ").strip()
        chat_id = input("Enter your chat ID: ").strip()
        print()
        
        if not bot_token or not chat_id:
            print("❌ Both token and chat ID required")
            return
        
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": "✅ <b>Test Successful!</b>\n\nYour Telegram bot is working correctly.",
                "parse_mode": "HTML"
            }
            
            print("Sending test message...")
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                print("✅ SUCCESS! Check your Telegram for the test message")
                print()
                print("Your credentials are correct:")
                print(f"  Bot Token: {bot_token[:20]}...")
                print(f"  Chat ID: {chat_id}")
                print()
                
                # Offer to save
                save = input("Save these credentials? (y/n): ").strip().lower()
                if save == 'y':
                    import json
                    config = {
                        "bot_token": bot_token,
                        "chat_id": chat_id,
                        "check_interval": 15
                    }
                    with open("bot_config.json", 'w') as f:
                        json.dump(config, f, indent=2)
                    print("✓ Saved to bot_config.json")
            else:
                print("❌ Failed to send message")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "3":
        # Run with sample data
        print("Running strategy with improved sample data...")
        print("This should generate trading signals")
        print()
        
        try:
            # Import the strategy
            from xauusd.xauusd_strategy import StrategyEngine, generate_sample_data, Backtester
            
            # Generate better sample data
            print("Generating realistic market data...")
            xauusd, dxy = generate_sample_data(periods=10000)
            xauusd.set_index('DateTime', inplace=True)
            dxy.set_index('DateTime', inplace=True)
            
            print(f"✓ Generated {len(xauusd)} bars")
            print()
            
            # Initialize strategy with more relaxed parameters
            print("Initializing strategy with test parameters...")
            strategy = StrategyEngine(
                ema_fast=20,
                ema_slow=50,
                ema_h4=200,
                zscore_period=20,
                zscore_entry=1.5,        # More lenient for testing
                zscore_exit=0.5,
                atr_period=14,
                sl_atr_multiplier=2.0,
                tp_atr_multiplier=3.0,
                sentiment_threshold=0.8   # More lenient for testing
            )
            
            print("Generating signals...")
            signals = strategy.generate_signals(xauusd, dxy)
            print()
            
            signal_count = (signals['Signal'] != 0).sum()
            print(f"✅ Generated {signal_count} trading signals!")
            print()
            
            if signal_count > 0:
                print("Running backtest...")
                trades, stats, equity = Backtester.run_backtest(signals)
                
                print()
                print("="*70)
                print("BACKTEST RESULTS")
                print("="*70)
                print(f"Total Trades:      {stats['total_trades']}")
                print(f"Win Rate:          {stats['win_rate']:.2f}%")
                print(f"Profit Factor:     {stats['profit_factor']:.2f}")
                print(f"Total Return:      {stats['total_return']:.2f}%")
                print("="*70)
                
                if stats['win_rate'] >= 60:
                    print("\n✅ Great! Strategy is working well!")
                    print("   You're ready to try with real MT5 data")
                else:
                    print("\n⚠️  Win rate below target")
                    print("   This is normal with sample data")
                    print("   Real market data should perform better")
            else:
                print("⚠️  No signals generated with test parameters")
                print("   This might mean the strategy is very selective")
                print("   Or sample data needs adjustment")
            
        except Exception as e:
            print(f"❌ Error running strategy: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "4":
        # Configure for live trading
        print("Setting up live trading configuration...")
        print()
        
        print("Step 1: Telegram Bot")
        print("-" * 40)
        bot_token = input("Bot Token (from @BotFather): ").strip()
        chat_id = input("Chat ID (from @userinfobot): ").strip()
        print()
        
        print("Step 2: Check Interval")
        print("-" * 40)
        interval = input("Minutes between checks [15]: ").strip() or "15"
        print()
        
        if bot_token and chat_id:
            import json
            config = {
                "bot_token": bot_token,
                "chat_id": chat_id,
                "check_interval": int(interval)
            }
            
            with open("bot_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✅ Configuration saved!")
            print()
            print("Testing connection...")
            
            try:
                import requests
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                data = {
                    "chat_id": chat_id,
                    "text": "✅ <b>Bot Configured!</b>\n\nReady to receive trading signals.",
                    "parse_mode": "HTML"
                }
                response = requests.post(url, data=data, timeout=10)
                
                if response.status_code == 200:
                    print("✅ Test message sent to Telegram!")
                    print()
                    print("Ready to start live trading:")
                    print("  python xauusd_strategy.py --live")
                else:
                    print("⚠️  Saved but couldn't send test message")
            except:
                print("⚠️  Saved but couldn't test connection")
        else:
            print("❌ Configuration cancelled")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")