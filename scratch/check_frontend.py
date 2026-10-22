import re

with open('frontend/src/App.tsx', encoding='utf-8') as f:
    text = f.read()

print('Length of App.tsx:', len(text))
tabs = re.findall(r"activeTab\s*===\s*['\"]([^'\"]+)['\"]", text)
print('Active tabs found:', sorted(list(set(tabs))))

# Find main sections / component headers
buttons = re.findall(r"<button[^>]*onClick=\{[^}]*setActiveTab\(['\"]([^'\"]+)['\"]", text)
print('Tab buttons found:', sorted(list(set(buttons))))
