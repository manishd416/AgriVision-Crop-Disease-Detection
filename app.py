    # Analyze button
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            predicted_class, confidence = predict_disease(model, uploaded_file)
            info = get_disease_info(predicted_class)

            # ⚠️ Low confidence check - STRONGER
            if confidence < 0.60:
                st.error(f"❌ Unable to classify (Confidence: {confidence*100:.1f}%)")
                st.warning("This image does not appear to be a Corn, Potato, or Tomato leaf. Please upload a clear leaf image of these crops only.")
                st.stop()  # Stop here, don't show fake results
            
            # Format disease name for display
            display_name = predicted_class.replace('___', ' - ')
            
            st.success("Analysis Complete!")
            st.markdown(f"### 🧬 Disease Detected: **{display_name}**")
            st.markdown(f"### 📊 Confidence: **{confidence*100:.2f}%**")
            
            # Progress bar for confidence
            st.progress(min(confidence, 1.0))
            
            st.markdown("---")
            st.subheader("📖 Disease Information")
            st.write(info["description"])
            
            st.subheader("💊 Treatment Recommendation")
            st.write(info["treatment"])
            
            st.info("⚠️ This is an AI-based suggestion. For accurate diagnosis, please consult an agricultural expert.")