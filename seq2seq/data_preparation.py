import pandas as pd
import json
# Sample data
data = [
    {"query": "I want to see waterfalls", "response": "Visit Ella for beautiful waterfalls"},
    {"query": "Where can I find beaches", "response": "You can go to Galle for beaches"},
    {"query": "What are some good places for hiking", "response": "Hiking is great in Knuckles Mountain Range"},
    {"query": "Tell me about wildlife safaris", "response": "You can visit Yala National Park"},
    {"query": "Where is a good place for surfing", "response": "Arugam Bay is great for surfing"},
    {"query": "Recommend a place for camping", "response": "You should try camping in Horton Plains"},
    {"query": "Where can we go to camping", "response": "You should visit Kandy for historical places"},
    {"query": "Can you suggest a place for swimming", "response": "You can visit Kitulgala for adventure sports"},
    {"query": "Tell me about places for scuba diving", "response": "You can visit Hikkaduwa for scuba diving"},
    {"query": "Recommend a place for snorkeling", "response": "You should try snorkeling in Hikkaduwa"},
    {"query": "What are some popular tourist destinations", "response": "You can visit Ella and Galle for tourism"},
    {"query": "Can you suggest a place for birdwatching", "response": "You can visit Sinharaja Forest Reserve"},
    {"query": "I want to see hiking trails", "response": "Visit Knuckles Mountain Range for great hiking"},
    {"query": "Where can I go for wildlife adventures", "response": "You can visit Yala National Park"},
    {"query": "Can you suggest a good place for sightseeing", "response": "You can visit Sigiriya for amazing views"},
    {"query": "Tell me about places for a scenic train ride", "response": "You can take the train from Kandy to Ella"},
    {"query": "Where is a good place for whale watching", "response": "You can visit Mirissa for whale watching"},
    {"query": "Where can I go for cultural experiences", "response": "You should visit Anuradhapura for cultural experiences"},
    {"query": "What is Sigiriya famous for?", "response": "Sigiriya is famous for its ancient rock fortress and stunning frescoes."},
    {"query": "How long does it take to climb Sigiriya Rock?", "response": "It typically takes about 1.5 to 2 hours to climb Sigiriya Rock."},
    {"query": "What should I bring when visiting Sigiriya?", "response": "Bring water, sunscreen, a hat, and comfortable shoes for the climb."},
    {"query": "Are there any entrance fees for Sigiriya?", "response": "Yes, there is an entrance fee to visit Sigiriya. Check the official website for current rates."},
    {"query": "Is there accommodation near Sigiriya?", "response": "Yes, there are several hotels and guesthouses near Sigiriya."},
    {"query": "Can I visit Sigiriya with children?", "response": "Yes, but be mindful of the climb and keep an eye on young children."},
    {"query": "What are the opening hours of Sigiriya?", "response": "Sigiriya is open from 7:00 AM to 5:30 PM."},
    {"query": "Are there restaurants or cafes near Sigiriya?", "response": "Yes, there are several restaurants and cafes nearby where you can have a meal."},
    {"query": "Is it crowded at Sigiriya?", "response": "Sigiriya can get crowded, especially during peak tourist season."},
    {"query": "What is the history of Sigiriya?", "response": "Sigiriya was built by King Kashyapa in the 5th century and is a UNESCO World Heritage Site."},
    {"query": "Can I hire a guide at Sigiriya?", "response": "Yes, you can hire a guide at the entrance for a more informative visit."},
    {"query": "What is the weather like in Sigiriya?", "response": "The weather in Sigiriya is generally hot and humid, so dress accordingly."},
    {"query": "Are there any nearby attractions to visit after Sigiriya?", "response": "Yes, you can visit the Dambulla Cave Temple and Minneriya National Park nearby."},
    {"query": "Is there a best time of day to visit Sigiriya?", "response": "Visiting early in the morning or late afternoon is ideal to avoid the midday heat."},
    {"query": "Can I see wildlife at Sigiriya?", "response": "Yes, you might see monkeys, birds, and other wildlife around Sigiriya."},
    {"query": "Are there any special events held at Sigiriya?", "response": "Occasionally, cultural events and performances are held near Sigiriya."},
    {"query": "What are the safety tips for climbing Sigiriya?", "response": "Wear sturdy shoes, stay hydrated, and be cautious of slippery steps, especially after rain."},
    {"query": "How can I learn more about the history of Sigiriya?", "response": "You can read the informational plaques on-site or hire a guide for detailed history."},
    {"query": "Is it a good idea to transfer straight to Tangalle from Colombo?", "response": "Yes, it's a good idea as it will be about a 3-hour drive on an excellent expressway from the airport."},
    {"query": "What should we do in Tangalle?", "response": "Relax on the beach and get over your jet lag. However, with a 9-year-old, you might prefer beaches with better swimming options like Unawatuna, Hikkaduwa, Weligama, or Mirissa."},
    {"query": "What activities are available in Ella?", "response": "Ella offers trekking, a zipline, and scenic train rides. It's a great place for some adventure."},
    {"query": "How do we travel from Ella to Kandy?", "response": "You can take a scenic train ride from Ella to Kandy. It's a beautiful journey through the hills."},
    {"query": "What should we see in Kandy?", "response": "In Kandy, visit the Temple of the Tooth, the botanical gardens, and consider a day trip to Lankatilleke, Embakke, and Pinnawela to see the elephants."},
    {"query": "How long should we stay near Sigiriya?", "response": "Stay two nights near Sigiriya to explore the Dambulla Temple, climb Sigiriya Rock, Pidurangala, and possibly Ritigala, and enjoy an elephant safari in Minneriya or Kaudulla."},
    {"query": "Is Wilpattu good for wildlife viewing?", "response": "Yes, Wilpattu is great for wildlife viewing, especially for seeing leopards, but sightings are not guaranteed. It's less crowded and offers a better experience than Yala."},
    {"query": "Should we visit Colombo?", "response": "Skip Colombo unless you have a specific reason to visit. Focus on other destinations."},
    {"query": "What is the best way to travel between these locations?", "response": "Hiring cars with drivers is convenient and allows you to enjoy the scenic routes and stop at various points of interest along the way."},
    {"query": "Are there any good dining options near the beaches?", "response": "For better dining options near the beach, consider staying closer to Galle, such as in Unawatuna, Hikkaduwa, Weligama, or Mirissa."},
    {"query": "What is the new attraction in Galle?", "response": "Sri Lanka’s first-ever underwater museum has been declared open in Galle."},
    {"query": "Who constructed the underwater museum in Galle?", "response": "The Sri Lanka Navy constructed the underwater museum in Galle."},
    {"query": "When was Sri Lanka’s first underwater museum opened?", "response": "Sri Lanka’s first-ever underwater museum was opened on 05th April 2020."},
    {"query": "Who opened the underwater museum in Galle?", "response": "Commander of the Navy, Vice Admiral Piyal De Silva, declared open the underwater museum."},
    {"query": "What makes the underwater museum in Galle special?", "response": "It is Sri Lanka’s first-ever underwater museum, adding a unique attraction for both local and foreign tourists."},
    {"query": "Where is Sri Lanka's first underwater museum located?", "response": "The underwater museum is located off the shores of Galle."},
    {"query": "Who is Vice Admiral Piyal De Silva?", "response": "Vice Admiral Piyal De Silva is the Commander of the Navy and a veteran in underwater diving."},
    {"query": "How was the underwater museum in Galle inaugurated?", "response": "The underwater museum was inaugurated by Vice Admiral Piyal De Silva cutting the ribbon."},
    {"query": "Why is the underwater museum in Galle significant?", "response": "It marks a milestone for Sri Lanka Navy and adds a unique underwater attraction to Galle, enhancing its appeal as a tourist destination."},
    {"query": "What can tourists expect at the underwater museum in Galle?", "response": "Tourists can explore various underwater exhibits constructed by the Sri Lanka Navy, making it a unique diving experience in Galle."},
    {"query": "How can I travel from Colombo to Jaffna?", "response": "You can take the intercity train from Colombo Fort to Jaffna. The train departs at 5:45 AM and arrives in Jaffna at 11:50 AM."},
    {"query": "Is the train from Colombo to Jaffna comfortable?", "response": "Yes, the air-conditioned first class is reasonably clean and comfortable, and the ticket is well-priced, less than US$ 10."},
    {"query": "What are the train timings from Jaffna to Colombo?", "response": "The train from Jaffna to Colombo Fort departs at 1:45 PM and arrives in Colombo at 8:05 PM."},
    {"query": "Can I book train tickets online for the Colombo to Jaffna route?", "response": "There is currently no online booking facility. You will need to book through a travel agent or have someone make the booking for you, and it is advisable to book well in advance."},
    {"query": "How can I book train tickets for the Colombo to Jaffna route?", "response": "You can book train tickets through a travel agent or someone local. It is advisable to book well in advance due to high demand."},
    {"query": "Is it safe to travel by train to Jaffna?", "response": "Yes, traveling by train to Jaffna is a safe and comfortable option, especially compared to long car journeys."},
    {"query": "What should I expect when taking the train to Jaffna?", "response": "Expect a comfortable journey with air-conditioned first-class seating. The train ride offers a scenic view and a chance to relax."},
    {"query": "What can I do in Jaffna?", "response": "In Jaffna, you can hire local guides to explore the area. There are many cultural and historical sites to visit, and it's recommended to plan a few activities while leaving space for serendipity."},
    {"query": "Are there travel agents who can help with booking train tickets to Jaffna?", "response": "Yes, travel agents can assist with booking train tickets. For example, Boutique Sri Lanka can help, although their prices might be higher than booking directly."},
    {"query": "How long should I stay in Jaffna?", "response": "Planning for 3 nights in Jaffna is a good idea to explore the region fully and experience its culture and attractions."},
    {"query": "How can I extend my visa in Sri Lanka?", "response": "To extend your visa in Sri Lanka, you need to visit the Immigration office in Colombo. You can download the application form from their official website."},
    {"query": "Where is the immigration office for visa extension in Sri Lanka?", "response": "The immigration office is in a government office complex, about 10km away from Colombo city center."},
    {"query": "Is it difficult to extend a visa in Sri Lanka?", "response": "With proper preparation and guidance, extending a visa in Sri Lanka is not complicated or costly."},
    {"query": "Can I extend my visa in the first few days of my arrival?", "response": "Yes, it is possible to extend your visa in the first few days in Colombo, but you should check if there are any new regulations implemented."},
    {"query": "What should I bring for my visa extension in Sri Lanka?", "response": "You should bring your passport, completed application form, and any other required documents as specified on the official website."},
    {"query": "Are there any guides available to help with the visa extension process?", "response": "Yes, some boutique guest houses near the immigration office in Colombo and local tour guides can assist you with the visa extension process."},
    {"query": "Can I find official information on visa extension in Sri Lanka online?", "response": "Official information on visa extension can be found on the Sri Lanka immigration official website, and you can also look for infographic guides online."},
    {"query": "How far is the immigration office from Colombo city center?", "response": "The immigration office is about 10km away from Colombo city center."},
    {"query": "Do I need to visit the immigration office in person for a visa extension?", "response": "Yes, you need to visit the immigration office in Colombo in person to extend your visa."},
    {"query": "Is there an online booking system for visa extensions in Sri Lanka?", "response": "No, there is currently no online booking system for visa extensions in Sri Lanka; you need to visit the immigration office in person."},
    {"query": "What are the best places to visit in Sri Lanka?", "response": "Some of the best places to visit in Sri Lanka include Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and Ella."},
    {"query": "Where should I go in Sri Lanka?", "response": "You should visit Sigiriya for its ancient rock fortress, Polonnaruwa for its historical ruins, Anuradhapura for its ancient city and sacred Bodhi tree, and Dambulla for its impressive cave temples."},
    {"query": "What are the must-see destinations in Sri Lanka?", "response": "Must-see destinations in Sri Lanka include Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and the hill country of Ella."},
    {"query": "What are the top tourist attractions in Sri Lanka?", "response": "Top tourist attractions in Sri Lanka are the Sigiriya rock fortress, the ancient city of Polonnaruwa, the sacred city of Anuradhapura, the Dambulla cave temple, and the Temple of the Tooth in Kandy."},
    {"query": "Which places should I not miss in Sri Lanka?", "response": "You should not miss Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and the scenic train ride to Ella."},
    {"query": "What are the top places to visit in Sri Lanka?", "response": "The top places to visit in Sri Lanka are Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and the beautiful hill station of Ella."},
    {"query": "Where are the best historical sites in Sri Lanka?", "response": "The best historical sites in Sri Lanka are Sigiriya, Polonnaruwa, Anuradhapura, and Dambulla."},
    {"query": "What are the best cultural sites in Sri Lanka?", "response": "The best cultural sites in Sri Lanka include the Temple of the Tooth in Kandy, the Sigiriya rock fortress, the ancient cities of Polonnaruwa and Anuradhapura, and the Dambulla cave temples."},
    {"query": "What are the best natural attractions in Sri Lanka?", "response": "The best natural attractions in Sri Lanka are the scenic hill country of Ella, the beaches of the southern coast, Horton Plains National Park, and the wildlife parks like Yala and Udawalawe."},
    {"query": "Where can I experience the best of Sri Lanka?", "response": "To experience the best of Sri Lanka, visit the historical sites of Sigiriya and Polonnaruwa, the cultural landmarks in Kandy and Dambulla, and the natural beauty of Ella and the southern beaches."},
    {"query": "What are the best places to visit in Sri Lanka?", "response": "Some of the best places to visit in Sri Lanka include Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and Ella."},
    {"query": "What are the must-see destinations in Sri Lanka?", "response": "Some of the best places to visit in Sri Lanka include Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and Ella."},
    {"query": "Where should I go in Sri Lanka?", "response": "Some of the best places to visit in Sri Lanka include Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and Ella."},
    {"query": "What are the top tourist attractions in Sri Lanka?", "response": "Some of the best places to visit in Sri Lanka include Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and Ella."},
    {"query": "Which places should I not miss in Sri Lanka?", "response": "Some of the best places to visit in Sri Lanka include Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and Ella."},
    {"query": "What are the top places to visit in Sri Lanka?", "response": "Some of the best places to visit in Sri Lanka include Sigiriya, Polonnaruwa, Anuradhapura, Dambulla, Kandy, and Ella."},
    {"query": "Where are the best historical sites in Sri Lanka?", "response": "Some of the best historical sites in Sri Lanka are Sigiriya, Polonnaruwa, Anuradhapura, and Dambulla."},
    {"query": "What are the best cultural sites in Sri Lanka?", "response": "Some of the best cultural sites in Sri Lanka include the Temple of the Tooth in Kandy, the Sigiriya rock fortress, the ancient cities of Polonnaruwa and Anuradhapura, and the Dambulla cave temples."},
    {"query": "What are the best natural attractions in Sri Lanka?", "response": "The best natural attractions in Sri Lanka are the scenic hill country of Ella, the beaches of the southern coast, Horton Plains National Park, and the wildlife parks like Yala and Udawalawe."},
    {"query": "Where can I experience the best of Sri Lanka?", "response": "To experience the best of Sri Lanka, visit the historical sites of Sigiriya and Polonnaruwa, the cultural landmarks in Kandy and Dambulla, and the natural beauty of Ella and the southern beaches."}


]  



#Convert to DataFrame
#df = pd.DataFrame(data)

#Save to CSV
#df.to_csv('chat_data.csv', index=False)

# Load data from JSON file
with open('chat_data.json', 'r') as json_file:
    data = json.load(json_file)

# Flatten the data into a DataFrame
df = pd.DataFrame([{"query": query, "response": item["response"]} for item in data for query in item["queries"]])

# Save to CSV
df.to_csv('chat_data.csv', index=False)