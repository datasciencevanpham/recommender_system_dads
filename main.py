
import streamlit as st
from config  import stop_words, df,df_clean,df_collab,df_review_full,top_products
from Functions import text_clean_2_model,dictionary, tfidf, index,recommender_id_model,recommender_text_model
import os, sys
import joblib
import random
from unidecode import unidecode
import pandas as pd
from streamlit_extras.dataframe_explorer import dataframe_explorer

# file edited by both

def format_price(price):
    # Chuyển đổi price thành một chuỗi số định dạng hàng ngàn
    price = float(price) # edit them
    formatted_price = "{:,.0f}".format(price)  # Định dạng số và thêm dấu phân cách hàng ngàn
    return formatted_price



# Define your display_product_info function (replace this with your product info display logic)
def display_product_info(product):
    # Customize this function to display product information
    st.write(f"Product Name: {product['product_name']}")
    #st.write(f"Rating: {product['customer_rating']}")
    formatted_price = format_price(product['price'])
    st.write(f"<b>Price: {formatted_price} VND<b>", unsafe_allow_html=True)

## edit mỗi hàng 3 sản phẩm
# # Define a function to display the latest products and information for a customer
# def display_latest_products_and_info(customer_id, excluded_product_ids=None):
#     # Filter the DataFrame for the selected customer, excluding the specified product IDs
#     customer_data = df_review_full[df_review_full['customer_id'] == customer_id]

#     # Drop duplicates based on customer and product, and sort by rating in descending order
#     unique_customer_products = customer_data.drop_duplicates(subset=['customer_id', 'product_name'])
#     latest_purchase = unique_customer_products.sort_values(by='customer_rating', ascending=False)

#     custom_image_width = 100  # Set the custom image width here

#     for _, product in latest_purchase.head(5).iterrows():
#         # Check if the product ID is in the excluded list; if not, display it
#         if excluded_product_ids is None or product['product_id'] not in excluded_product_ids:
#             # Customize this part to display the product information as needed
#             st.image(product['image'], width=custom_image_width, output_format='JPEG')
#             st.write(f"Rating: {product['customer_rating']}")
#             formatted_price = format_price(product['price']) #st.write(f"<b>Price: {format_price(product['price'])} VND</b>"
#             st.write(f"<b>Price: {formatted_price} VND<b>", unsafe_allow_html=True)

#             # Add an empty line to separate products
#             st.write("")

# Define a function to display the latest products and information for a customer
def display_latest_products_and_info(customer_id, excluded_product_ids=None):
    # Filter the DataFrame for the selected customer, excluding the specified product IDs
    customer_data = df_review_full[df_review_full['customer_id'] == customer_id]

    # Drop duplicates based on customer and product, and sort by rating in descending order
    unique_customer_products = customer_data.drop_duplicates(subset=['customer_id', 'product_name'])
    latest_purchase = unique_customer_products.sort_values(by='customer_rating', ascending=False)

    # Create a container for the latest product images and information
    cols = st.columns(3)

    # Display the latest products and information
    for i, (_, product) in enumerate(latest_purchase.head(5).iterrows()):
        # Check if the product ID is in the excluded list; if not, display it
        if excluded_product_ids is None or product['product_id'] not in excluded_product_ids:
            with cols[i % 3]:
                st.image(product['image'], width=100, output_format='JPEG')
                st.write(f"Rating: {product['customer_rating']}")
                formatted_price = format_price(product['price'])
                st.write(f"<b>Price: {formatted_price} VND<b>", unsafe_allow_html=True)

            # Add an empty line to separate products
            if i % 3 == 2:
                st.write("")
                ### end edit


# Define the Streamlit app function

def run_recommender_app_collab():
    st.write("#### Collaborative Filtering Recommendation")
    
    # Allow the user to enter a customer name or ID
    customer_input = st.text_input("Enter a Customer Name or a Customer ID")
    
    # Define a list of available options for the number of products
    num_products_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Allow the user to select the number of products to display
    selected_option = st.selectbox("Select the number of products to display:", num_products_options, index=num_products_options.index(10))
    
    # Initialize the list of excluded product IDs
    excluded_product_ids_recommendations = []
   
    show_random_customer_ids = False

    if st.button("Find Recommendations"):
        if customer_input:
            if customer_input.isdigit():
                customer_id = int(customer_input)
                if customer_id > 0:
                    # Handle the case when input is a numeric customer ID
                    st.subheader(f"The latest purchase of Customer ID {customer_id}:")
                    excluded_product_ids_latest = []  # Initialize for display_latest_products_and_info
                    display_latest_products_and_info(customer_id, excluded_product_ids_latest)
                    
                    st.subheader(f"Top {selected_option} Recommended Products for Customer ID {customer_id} (excluding latest):")

                    # Retrieve recommendations using the cached function
                    recommendations = df_collab[df_collab['customer_id'] == customer_id].drop_duplicates(subset='product_id').head(selected_option)

                    if not recommendations.empty:
                        # # Create a container for the product images
                        # col1, col2, col3 = st.columns(3)

                        # # Add some CSS to control the spacing between images
                        # st.write(
                        #     """
                        #     <style>
                        #     .stImage {
                        #         margin-right: 20px; /* Adjust the right margin as needed */
                        #     }
                        #     </style>
                        #     """,
                        #     unsafe_allow_html=True,
                        # )

                        # for i, (_, product) in enumerate(recommendations.iterrows(), start=1):
                        #     # Display product information in a column
                        #     with col1:
                        #         st.image(product['image'], use_column_width=True, output_format='JPEG')
                        #         st.write(f"Customer ID: {customer_id}")  # Display customer ID
                        #         display_product_info(product)

                        #     # Add the product ID to the excluded list
                        #     excluded_product_ids_recommendations.append(product['product_id'])

                        #     # Start a new row every three products
                        #     if i % 3 == 0:
                        #         col1, col2, col3 = st.columns(3)
                        ### edit new code
                        # Create a container for the product images
                        cols = st.columns(3)

                        # Add some CSS to control the spacing between images
                        st.write(
                            """
                            <style>
                            .stImage {
                                margin-right: 20px; /* Adjust the right margin as needed */
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        for i, (_, product) in enumerate(recommendations.iterrows(), start=1):
                            # Display product information in a column
                            with cols[i % 3]:
                                st.image(product['image'], use_column_width=True, output_format='JPEG')
                                st.write(f"Customer ID: {customer_id}")  # Display customer ID
                                display_product_info(product)

                                if 'description' in product and product['description']:
                                    # Create an expander widget to show more details about the product
                                    with st.expander("Read more", expanded=False):
                                        st.write(product['description'])

                            # Add the product ID to the excluded list
                            excluded_product_ids_recommendations.append(product['product_id'])
                            if 'description' in product and product['description']:
                                excluded_product_ids_recommendations.append(product['description'])

                            # Start a new row every three products
                            if i % 3 == 0:
                                cols = st.columns(3)
                                ##### end edit

                else:
                    st.warning("Please enter a valid numeric customer ID.")
            else:
                # Handle the case when input is a customer name
                customer_name = customer_input
                customer_input_lower = customer_name.lower()  # Convert input to lowercase
                customer_input_clean = unidecode(str(customer_input_lower))
                # Check if 'customer_name' is a string before applying 'lower()'
                recommendations = df_collab[df_collab['customer_name'].apply(lambda x: isinstance(x, str) and x.lower().find(customer_input_clean) != -1)].sort_values(by='customer_rating', ascending=False).head(selected_option)

                if not recommendations.empty:
                    # # Create a container for the product images
                    # col1, col2, col3 = st.columns(3)

                    # # Add some CSS to control the spacing between images
                    # st.write(
                    #     """
                    #     <style>
                    #     .stImage {
                    #         margin-right: 20px; /* Adjust the right margin as needed */
                    #     }
                    #     </style>
                    #     """,
                    #     unsafe_allow_html=True,
                    # )

                    # for i, (_, product) in enumerate(recommendations.iterrows(), start=1):
                    #     # Display product information in a column
                    #     with col1:
                    #         st.image(product['image'], use_column_width=True, output_format='JPEG')
                            
                    #         # Check if customer_name is a string before applying lower()
                    #         customer_name = product['customer_name']
                    #         if isinstance(customer_name, str):
                    #             customer_name = customer_name.lower()
                            
                    #           # Display customer name
                    #         display_product_info(product)

                    #     # Add the product ID to the excluded list
                    #     excluded_product_ids_recommendations.append(product['product_id'])

                    #     # Start a new row every three products
                    #     if i % 3 == 0:
                    #         col1, col2, col3 = st.columns(3)

                    ## edit code
                    # Create a container for the product images
                    cols = st.columns(3)

                    # Add some CSS to control the spacing between images
                    st.write(
                        """
                        <style>
                        .stImage {
                            margin-right: 20px; /* Adjust the right margin as needed */
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Generate a list of products for each row
                    products_per_row = [recommendations.iloc[i:i+3] for i in range(0, len(recommendations), 3)]

                    # Loop through each row and display the products in columns
                    for i, row in enumerate(products_per_row):
                        with cols[0]:
                            if 0 < i < len(products_per_row):
                                st.write('---')
                        with cols[1]:
                            if 0 < i < len(products_per_row):
                                st.write('---')
                        with cols[2]:
                            if 0 < i < len(products_per_row):
                                st.write('---')

                        # Loop through each product in the row and display it
                        for _, product in row.iterrows():
                            with cols[_ % 3]:
                                st.image(product['image'], use_column_width=True, output_format='JPEG')

                                # Check if customer_name is a string before applying lower()
                                customer_name = product['customer_name']
                                if isinstance(customer_name, str):
                                    customer_name = customer_name.lower()
                                
                                # Display customer name
                                display_product_info(product)

                                # Create an expander widget to show more details about the product
                                with st.expander("Read more", expanded=False):
                                    st.write(product['description'])
                                
                            # Add the product ID to the excluded list
                            excluded_product_ids_recommendations.append(product['product_id'])
      
                            if product['description']:
                                excluded_product_ids_recommendations.append(product['description'])
                                ##### end edit

        else:
            st.warning("Please enter a customer name or ID")
    
    # Add a "Find Another Customer" button to reset the page
    if st.button("Find Another Customer"):
        st.experimental_rerun()
    else:
        # Suggest 10 random customer IDs # edit
        # random_customer_ids = random.sample(df_collab['customer_id'].tolist(), 10)
        st.write("You can choose randomly customer IDs in the table below:")
        # st.write(random_customer_ids)
        unique_customer_ids = df_collab['customer_id'].unique()
        unique_customer_ids = pd.DataFrame({'customer_id': unique_customer_ids}).sort_values('customer_id')
        st.dataframe(dataframe_explorer(unique_customer_ids[['customer_id']].astype("str")),use_container_width=True, hide_index=True)  
        
        
        # Suggest 10 random customer names # edit
        # random_customer_names = random.sample(df_collab['customer_name'].tolist(), 10)
        st.write("You can choose randomly customer names in the table below:")
        # st.write(random_customer_names)
        unique_customer_name = df_collab['customer_name'].unique()
        unique_customer_name = pd.DataFrame({'customer_name': unique_customer_name}).sort_values('customer_name')
        st.dataframe(dataframe_explorer(unique_customer_name[['customer_name']].astype("str")),use_container_width=True, hide_index=True)




def display_product_info_2(product):
    # Customize this function to display the product information as you need
    st.write(f"Name: {product['product_name']}")
    st.write(f"Product ID: {product['product_id']}")
    #st.write(f"Rating: {product['rating']}")
    formatted_price = format_price(product['price'])
    st.write(f"<b>Price: {formatted_price} VND<b>", unsafe_allow_html=True)
    # Create an expander for the description
    with st.expander("Read More"):
        st.write(product['description'])






def display_top_products(top_products):
    st.write("##### Top 5 Most Popular Products")
    for _, product in top_products.iterrows():
        product_name = product['product_name']
        image = product['image']
        price = product['price']
        formatted_price = format_price(price)

        # Use HTML to display product information horizontally with a smaller image, product name, and price
        st.markdown(f"<div style='display: flex; flex-direction: row; align-items: center;'>"
                        f"<div style='flex: 1;'>"
                        f"{product_name}"
                        f"<br><b>Price: {formatted_price} VND</b>"
                        f"</div>"
                        f"<div style='flex: 1;'><img src='{image}' alt='{product_name}' width='100'></div>"
                        f"</div>", unsafe_allow_html=True)
    st.write("----")
# Call the function to display the top 5 products by count
#display_top_products_by_id(5)




def run_contend_based_recommender_app(choice):
    if choice == 'Content-Based Recommendation':
        st.write("#### Content-Based Recommendation")
    
        # Allow the user to enter a product ID or name
        product_input = st.text_input("Enter a Product ID or Product Name:")

        # Define a list of available options for the number of products
        num_products_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Allow the user to select the number of products to display
        selected_option = st.selectbox("Select the number of products to display:", num_products_options)

        if st.button("Find Recommendations"):
            if product_input:
                if product_input.isdigit():
                    product_id = int(product_input)  # Convert to integer

                    # Retrieve product information for the searched product
                    info_id_search = df[df['product_id'] == product_id].iloc[0]  # Get the first row as it contains the searched product

                    # Replace this with your actual recommendation logic
                    result_id_search = recommender_id_model(product_id, dictionary, tfidf, index)

                    if not info_id_search.empty:
                        # Display product information for the searched product
                        st.subheader("Your searched product:")
                        st.image(info_id_search['image'], use_column_width=70, output_format='JPEG')
                        display_product_info_2(info_id_search)

                        if not result_id_search.empty:
                            # Filter out products with duplicate IDs
                            result_id_search = result_id_search.drop_duplicates(subset=['product_id'])

                            # Calculate the number of products to display
                            num_to_display = min(selected_option, len(result_id_search))
                            
                            ######## edit vì ko hiển thị đủ 3 cột 
                            # # Display recommended products
                            # st.subheader("Recommended Products:")
                            # col1, col2, col3 = st.columns(3)

                            # # Add some CSS to control the spacing between images
                            # st.write(
                            #     """
                            #     <style>
                            #     .stImage {
                            #         margin-right: 20px; /* Adjust the right margin as needed */
                            #     }
                            #     </style>
                            #     """,
                            #     unsafe_allow_html=True,
                            # )

                            # i = 0  # Counter for displayed products
                            # while i < num_to_display:  # Display only the specified number of products
                            #     product = result_id_search.iloc[i]
                            #     # Display product information in a column
                            #     with col1:
                            #         st.image(product['image'], use_column_width=70, output_format='JPEG')
                            #         display_product_info_2(product)

                            #     # Increment the counter
                            #     i += 1

                            #     # Start a new row every three products
                            #     if i % 3 == 0:
                            #         col1, col2, col3 = st.columns(3)

                            # # Update selected_option to match the number of products displayed
                            # selected_option = num_to_display

                            # 

                        ######## new code
                            # Display recommended products
                            st.subheader("Recommended Products:")
                            col1, col2, col3 = st.columns(3)

                            # Add some CSS to control the spacing between images
                            st.write(
                                """
                                <style>
                                .stImage {
                                    margin-right: 20px; /* Adjust the right margin as needed */
                                }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )

                            i = 0  # Counter for displayed products
                            while i < num_to_display:  # Display only the specified number of products
                                product = result_id_search.iloc[i]
                                # Display product information in a column
                                with col1:
                                    st.image(product['image'], use_column_width=70, output_format='JPEG')
                                    display_product_info_2(product)

                                # Increment the counter
                                i += 1

                                # Check if there are still products to display
                                if i >= num_to_display:
                                    break

                                product = result_id_search.iloc[i]
                                with col2:
                                    st.image(product['image'], use_column_width=70, output_format='JPEG')
                                    display_product_info_2(product)

                                i += 1

                                # Check if there are still products to display
                                if i >= num_to_display:
                                    break

                                product = result_id_search.iloc[i]
                                with col3:
                                    st.image(product['image'], use_column_width=70, output_format='JPEG')
                                    display_product_info_2(product)

                                i += 1

                                # Check if it's time to start a new row
                                if i % 3 == 0 and i < num_to_display:
                                    col1, col2, col3 = st.columns(3)

                            # Update selected_option to match the number of products displayed
                            selected_option = num_to_display
                            ######## end new code
                        else:
                            st.info("No recommendations available for the selected product.")
                    else:
                        st.info("No information available for the selected product ID.")
                else:
                    product_name = product_input
                    # Implement content-based recommendation
                    content_based_results = recommender_text_model(product_name, dictionary, tfidf, index, stop_words)

                    if not content_based_results.empty:
                        # Filter out products with duplicate IDs
                        content_based_results = content_based_results.drop_duplicates(subset=['product_id'])

                        # Calculate the number of products to display
                        num_to_display = min(selected_option, len(content_based_results))

                        st.subheader(f"Top {num_to_display} Recommended Products for Product Name: {product_name}")
                        ############ edit vì ko hiển thị đủ 3 cột 
                        # # Display recommended products
                        # col1, col2, col3 = st.columns(3)

                        # # Add some CSS to control the spacing between images
                        # st.write(
                        #     """
                        #     <style>
                        #     .stImage {
                        #         margin-right: 20px; /* Adjust the right margin as needed */
                        #     }
                        #     </style>
                        #     """,
                        #     unsafe_allow_html=True,
                        # )

                        # i = 0  # Counter for displayed products
                        # while i < num_to_display:  # Display only the specified number of products
                        #     product = content_based_results.iloc[i]
                        #     # Display product information in a column
                        #     with col1:
                        #         st.image(product['image'], use_column_width=70, output_format='JPEG')
                        #         display_product_info_2(product)

                        #     # Increment the counter
                        #     i += 1

                        #     # Start a new row every three products
                        #     if i % 3 == 0:
                        #         col1, col2, col3 = st.columns(3)
                        #   ##########      

                        # # Update selected_option to match the number of products displayed
                        # selected_option = num_to_display

                        ####### new code
                        # Display recommended products
                        st.subheader("Recommended Products:")
                        col1, col2, col3 = st.columns(3)

                        # Add some CSS to control the spacing between images
                        st.write(
                            """
                            <style>
                            .stImage {
                                margin-right: 20px; /* Adjust the right margin as needed */
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        i = 0  # Counter for displayed products
                        while i < num_to_display:  # Display only the specified number of products
                            product = content_based_results.iloc[i]
                            # Display product information in a column
                            with col1:
                                st.image(product['image'], use_column_width=70, output_format='JPEG')
                                display_product_info_2(product)

                            # Increment the counter
                            i += 1

                            # Check if there are still products to display
                            if i >= num_to_display:
                                break

                            product = content_based_results.iloc[i]
                            with col2:
                                st.image(product['image'], use_column_width=70, output_format='JPEG')
                                display_product_info_2(product)

                            i += 1

                            # Check if there are still products to display
                            if i >= num_to_display:
                                break

                            product = content_based_results.iloc[i]
                            with col3:
                                st.image(product['image'], use_column_width=70, output_format='JPEG')
                                display_product_info_2(product)

                            i += 1

                            # Check if it's time to start a new row
                            if i % 3 == 0 and i < num_to_display:
                                col1, col2, col3 = st.columns(3)

                        # Update selected_option to match the number of products displayed
                        selected_option = num_to_display
                        ###### end new code

                        # Create a dropdown menu with related product names
                        related_product_names = content_based_results['product_name'].tolist()

                        # Display the related product names as buttons
                        selected_related_product_name = st.selectbox("Select a Related Product:", related_product_names, key="related_product_selectbox")

                        # Use the selected product name to set the new product_input
                        if selected_related_product_name:
                            product_input = selected_related_product_name

                    else:
                        st.info("No content-based recommendations available for the entered product name.")

                # Add a button to enter another product
                if st.button("Enter Another Product"):
                    product_input = st.text_input("Enter a Product ID or Name", key='product_input')  # Reset the input field
            else:
                st.warning("Please enter a Product ID or Product Name.")
        else:
            # Suggest 10 random customer IDs # edit
            # random_product_ids = random.sample(df['product_id'].tolist(), 10)
            # st.write("Randomly suggested Product IDs:")
            # st.write(random_product_ids)
            st.write("You can choose randomly a Product ID from 'product_id' table below:")
            st.dataframe(dataframe_explorer(df[['product_id']].astype("str")),use_container_width=True)



# Main Streamlit app
if __name__ == "__main__":
    st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 2.5rem;">Data Science - Project 2</h1>
        <p style="font-size: 1.5rem;">Recommender System</p>
    </div>
    """,
    unsafe_allow_html=True,
    )
   
    
    menu = ["Business Objectives", "Content-based filtering overview", "Content-Based Recommendation","Collaborative Filtering overview","Collaborative Filtering Recommendation"]
    choice = st.sidebar.selectbox('Menu', menu)
    
    if choice == 'Business Objectives':
        st.subheader("Business Objectives")
        st.write("""
        Recommender systems are used by E-commerce sites to suggest products to their customers. The products can be recommended based on the top overall sellers on a site, based on the demographics of the customer, or based on an analysis of the past buying behavior of the customer as a prediction for future buying behavior.
        """)  
        st.write("""
        Problem/Requirement: In this project, data is collected from an e-commerce website, assuming that the website does not have a recommender system. The objective is to build a Recommendation System to suggest and recommend products to users/customers. The goal is to create recommendation models, including Content-based filtering and Collaborative filtering, for one or multiple product categories on **tiki.vn**, providing personalized choices for users.
        """)
        st.image("content.jpg")
        st.image("Content_filtering.png")
        st.image("collaborative.jpg")
        st.image("Collaborative_filtering.png")
        st.write("**Group members:**")
        st.write("* Nguyễn Thị Huyền Trang.")
        st.write("* Phạm Thanh Vân.")

    elif choice == 'Content-based filtering overview':
        st.write("#### Content-based Filtering Project Overview")
        st.write("##### 1. Product Overview")
        # st.dataframe(df[["product_id", "product_name","price","rating","description","brand","group"]].head(3)) # edit
        st.dataframe(dataframe_explorer(df[["product_id", "product_name","price","rating","description","brand","group"]].astype("str")),use_container_width=True) 
        st.write("###### Data before cleaning includes the combination of name, description, brand, and group")
        # st.dataframe(df[["product_id", "description"]].head(3)) # edit
        st.dataframe(dataframe_explorer(df[["product_id", "description"]].astype("str")),use_container_width=True) 
        st.write("###### Data after cleaning and tokenization with 'Underthesea' ")
        # st.dataframe(df_clean[["product_id", "description_ws"]].head(3))  
        st.dataframe(dataframe_explorer(df_clean[["product_id", "description_ws"]].astype("str")),use_container_width=True)

        st.write("##### 2. Build Content-based filtering with Gensim")
        st.write("This is an explanation of how our product recommendation system works.")

        # Step 1: Cleaning and Tokenizing Text
        st.write("**Step 1: Cleaning and Tokenizing Text with underthesea**")
        st.write("In the first crucial step of our recommendation system, we meticulously clean and tokenize the product descriptions using the advanced 'underthesea' natural language processing technology. This intricate process ensures that each product description is transformed into a collection of individual words, ready to undergo further analysis and scrutiny")

        # Step 2: Creating a Dictionary and Corpus
        st.write("**Step 2: Creating a Dictionary and Corpus**")
        st.write("Following the initial cleaning process, we embark on the journey of creating a comprehensive dictionary and corpus. This step involves meticulously cataloging all the unique words encountered in the product descriptions. Furthermore, we painstakingly count the occurrences of each word in every product description. This meticulous record-keeping enables us to gain valuable insights into the linguistic richness of our product library.")

        # Step 3: TF-IDF Transformation
        st.write("**Step 3: TF-IDF Transformation**")
        st.write("As we delve deeper into the heart of our recommendation system, we harness the power of TF-IDF (Term Frequency-Inverse Document Frequency) transformation. This transformation is instrumental in identifying the most salient and impactful words within each product description. By assigning weights to each word based on its significance, we can pinpoint the words that define a product's uniqueness and appeal.")

        # Step 4: Calculating Similarities
        st.write("**Step 4: Calculating Similarities**")
        st.write("In this pivotal phase, we embark on the mathematical journey of calculating similarities between products. Using the well-established cosine similarity metric, we unravel the hidden relationships between products. By measuring the cosine of the angle between product vectors, we gain insights into how similar or dissimilar each product is to others in our extensive library.")

        # Step 5: Finding Recommendations
        st.write("**Step 5: Finding Recommendations**")
        st.write(" The core of our recommendation system lies in finding the most relevant product recommendations for each item in our collection. To achieve this, we employ a sophisticated algorithm that meticulously analyzes the product similarities we've uncovered. For each product, we painstakingly identify the top 5 most similar products as recommendations, ensuring that our suggestions are highly tailored to your preferences.")

        # Step 6: Displaying Recommendations
        st.write("**Step 6: Displaying Recommendations**")
        st.write("As a culmination of our extensive efforts, we present you with the cream of the crop—our meticulously curated recommendations. These handpicked products have earned their place as the top recommended items in our library. With a keen eye for quality and relevance, we ensure that your shopping experience is nothing short of exceptional.")

        # Conclusion
        st.write("Our recommendation system acts as your trusted shopping companion, resembling a helpful librarian in the digital realm. By suggesting products based on their descriptions and inherent characteristics, we aim to enhance your shopping journey, making it more enjoyable and tailored to your unique preferences.")

    elif choice=='Collaborative Filtering overview':
        st.write("#### Collaborative Filtering Project Overview")
        st.write("##### 1. Dataset Overview")
        # st.dataframe(df_review_full.head(3)) # edit
        st.dataframe(dataframe_explorer(df_review_full[['customer_id', 'product_id', 'customer_rating', 'product_name', 'brand', 'price', 'image']].astype("str")),use_container_width=True)  
        st.write("Selecting necessary attributes for analysis")
        # st.dataframe(df_review_full[["customer_id","product_id","customer_rating"]].head(3))
        st.dataframe(dataframe_explorer(df_review_full[["customer_id","product_id","customer_rating"]].astype("str")),use_container_width=True)
        st.write("##### 2. Build Collaborative filtering with ALS algorithm")
        st.write("**Step 1: Data Preparation**")         
        st.write("In this project, the dataset was collected and cleaned to ensure it contained user interactions with products, such as ratings and purchase histories. Missing values and outliers were handled to ensure data quality.")
        st.write("**Step 2: Feature Engineering**")
        st.write("To make the data suitable for modeling, categorical user and product identifiers were converted into numerical indices using techniques like StringIndexer. This transformation allowed for the mathematical operations required for collaborative filtering.")
        st.write("**Step 3: Train-Test Split**")
        st.write("The dataset was divided into a training set and a test set to accurately assess the model's performance. This division enabled the model to be trained on one portion of the data while reserving the other for evaluating how well recommendations were made on unseen interactions.")
        st.write("**Step 4: ALS Model**")
        st.write("The ALS (Alternating Least Squares) algorithm was chosen for building the collaborative filtering recommendation system. Hyperparameters, such as the rank of latent factors and regularization strength, were tuned to optimize the model's performance.")
        st.write("**Step 5: Model Training**")
        st.write("With the ALS algorithm and optimal hyperparameters in place, the recommendation model was trained on the training dataset. During training, the ALS model learned latent factors representing users and products, aiming to minimize the error in predicting user-item interactions.")
        st.write("**Step 6: Evaluation and Deployment**")
        st.write("After training, the model's performance was evaluated using the test dataset. Metrics  RMSE (Root Mean Squared Error)was used to measure how well the model predicted user preferences. Once the model's accuracy was deemed satisfactory, it was deployed to provide personalized recommendations to users, enhancing their experience on the platform.")
        st.write("Following these steps, a collaborative filtering recommendation system with ALS was successfully implemented, providing users with relevant product suggestions based on their interactions and preferences.")
    elif choice == 'Content-Based Recommendation':
        run_contend_based_recommender_app(choice)
        display_top_products(top_products)
        
    elif choice == 'Collaborative Filtering Recommendation':
        run_recommender_app_collab()





